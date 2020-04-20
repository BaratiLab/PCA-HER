import threading
import numpy as np
#import sklearn.datasets, sklearn.decomposition
from baselines.der.util import store_args
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import itertools
import time
import copy


class HVBuffer:
    @store_args
    def __init__(self, buffer_shapes, size_in_transitions, hv_successful_only = False, 
                 hv_evar = 0.9, hv_shape = 'circle', hv_stdevs = 3, hv_rung_enabled=True,
                 hv_collect_outliers = True):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        assert type(self.hv_stdevs)==int, 'The number of standard deviations should be an integer.'
        self.buffer_shapes['o_2'] = self.buffer_shapes['o']
        self.buffer_shapes['ag_2'] = self.buffer_shapes['ag']
        
        self.size = size_in_transitions #// T

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        
        self.buffers = {key: np.empty([self.size, shape[-1]])
                        for key, shape in self.buffer_shapes.items()}
        
        # hv buffer reshaping to ignore episode structure
        #print('[__init__@hv_buffer] -> Previous HV Buffer Shape for Observations:',self.buffers['o'].shape)
        for key, value in self.buffers.items():
            columns = value.shape[-1]
            self.buffers[key] = value.reshape(-1,columns)
        #print('[__init__@hv_buffer] -> New HV Buffer Shape for Observations:',self.buffers['o'].shape)
        
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, hv_batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # make assumption reward is always zero (good)
        # Current size = buffers['g'].shape[0]
        buffers['r'] = np.array([0]*self.current_size, dtype=np.float32)
        #buffers['info_is_success'] = np.array([[1]]*self.current_size,dtype=np.float32)
        
        # the state and goal at t+1 if undefined
        #if 'o_2' not in buffers: buffers['o_2'] = buffers['o']#[:, 1:, :]
        #if 'ag_2' not in buffers: buffers['ag_2'] = buffers['ag']#[:, 1:, :]
            
        # sample transitions randomly
        # we need a [] of numbers between 0 and current_size
        hv_indices = np.random.choice(self.current_size, hv_batch_size, replace=False)
        transitions = {key:value[hv_indices] for key,value in buffers.items()}
    
        # check transitions data is complete
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions
    
    def store_buffer(self, buffer):
        """buffer is a filtered dictionary of high-value transitions
        """
        # Clear the buffer
        
        buffer = buffer.copy()
        
        # Making sure the goal is also the achieved goal, HER trick
        buffer['g'] = buffer['ag']
        
        # reshape buffer if necessary
        if len(buffer[list(buffer.keys())[0]].shape) > 2:
            if 'o_2' not in buffer: buffer['o_2'] = buffer['o'][:, 1:, :]
            if 'ag_2' not in buffer: buffer['ag_2'] = buffer['ag'][:, 1:, :]
            for key, value in buffer.items():
                columns = value.shape[-1]
                buffer[key] = value.reshape(-1,columns)
        
        # size of buffer is number of entries
        buffer_size = max(sum(buffer[list(buffer.keys())[0]] !=0)) #.shape[0]
        assert buffer_size == buffer[list(buffer.keys())[0]].shape[0], 'Zeros found in HV Buffer. Please check storing and filtering algorithms.'
        
        # Making all transitions successful, HER trick
        buffer['info_is_success'] = np.array([[1]]*buffer_size,dtype=np.float32)
        
        with self.lock:
            self.current_size = 0
            
            # get indexes on buffer. Sequential at first, later random
            idxs = self._get_storage_idx(buffer_size)
            idxs_indices = np.arange(len(idxs))
            
            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = buffer[key][idxs_indices] #[idxs]
            
            self.n_transitions_stored += buffer_size
    
    def clean_buffer(self, buffer): #make it 2D
        bufferER = copy.deepcopy(buffer)
        # 1 - Add o_2 and ag_2 to ER buffer copy
        if len(bufferER['o'].shape) > 2: 
            bufferER['o_2'] = bufferER['o'][:, 1:, :]
            bufferER['o'] = bufferER['o'][:,:-1,:] # reshape ['o'] and ['ag'] from 60 to 59
        
        if len(bufferER['ag'].shape) > 2: 
            bufferER['ag_2'] = bufferER['ag'][:, 1:, :] #1 step into the future
            bufferER['ag']=  bufferER['ag'][:,:-1,:] # reshape ['o'] and ['ag'] from 60 to 59
        

        # 2 - Reshape ER buffer to get rid of episodes & remove zeros
        def bufferReshape(bufferER):
            info_shape = 0
            for i, key in enumerate(bufferER.keys()):
                shape_ = bufferER[key].shape[-1]
                rshp = np.reshape(bufferER[key], (-1,shape_))
                if 'info_' not in key:
                    rshp = rshp[~np.all(rshp == 0, axis=1)]
                    if i == 0: info_shape = rshp.shape[0]
                else:
                    rshp = rshp[:info_shape]
                bufferER[key] = rshp
            return bufferER
        
        bufferER = bufferReshape(bufferER)

        if self.hv_successful_only:
            success_info = bufferER['info_is_success']
            indices = [i for i in range(len(success_info)) if success_info[i][0] == 1]
            for key in bufferER.keys():
                bufferER[key] = np.take(bufferER[key],indices,axis=0)
        else: pass
        return bufferER
    
    # Standardized Data
    def standardize_data(self, mydarray):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(mydarray)
        return scaled

    def run_PCA(self, scaled):
        pca8 = PCA(n_components=self.hv_evar) #,random_state = 1
        principalCp = pca8.fit_transform(scaled)
        print('Number of components observed:', principalCp.shape[1])
        return principalCp

    # def get_boundary_pt_indices(self,principalCp,stdevs,out_bound_type):
    #     cols = list(np.arange(principalCp.shape[1]))
    #     bound_point = []
    #     biglst = []

    #     for a,b in itertools.combinations(cols,2):
    #         if a != b:
    #             #plt.title(str(a+1)+" vs "+str(b+1))
    #             #plt.scatter(principalCp[:,a],principalCp[:,b]); #plt.show();
    #             combo = np.vstack((principalCp[:,a],principalCp[:,b]))
    #             combo = np.transpose(combo)
    #             biglst.append(combo) 

    #     for i in range(len(biglst)):
    #         points = np.array(biglst[i])
    #         if out_bound_type == 'circle':
    #             indices = ((np.square((points[:,0])) + np.square(points[:,1])) < (np.square(points[:,0].mean() + points[:,0].std() * stdevs) + np.square(points[:,1].mean() + points[:,1].std() * stdevs)))
    #         else:
    #             indices = (points[:,0] > points[:,0].mean() + points[:,0].std() * stdevs) | (points[:,0] < points[:,0].mean() - points[:,0].std() * stdevs) | (points[:,1] > points[:,1].mean() + points[:,0].std() * stdevs) | (points[:,1] < points[:,1].mean() - points[:,0].std() * stdevs)
    #         inliners = points[~indices]
    #         if len(inliners) < 3: continue 
    #         hull = ConvexHull(inliners)       
    #         for simplex in hull.simplices:
    #             #plt.plot(inliners[simplex, 0], inliners[simplex, 1], 'k-') 
    #             bound_point.append(simplex) 
    #         #plt.show(); 
            
    #     bd = np.concatenate(bound_point).ravel().tolist()
    #     indices_bounds=np.unique(bd)
    #     print(indices_bounds)
    #     return indices_bounds
    
    # rings
    
    def _points_indices(self, points, j = None):
        j = j or self.hv_stdevs
        if self.hv_shape == 'circle':
           indices = ((np.square((points[:,0])) + np.square(points[:,1])) > (np.square(points[:,0].mean() + points[:,0].std() * j) + np.square(points[:,1].mean() + points[:,1].std() * j)))
        else:
           indices = (points[:,0] > points[:,0].mean() + points[:,0].std() * j) | (points[:,0] < points[:,0].mean() - points[:,0].std() * j) | (points[:,1] > points[:,1].mean() + points[:,1].std() * j) | (points[:,1] < points[:,1].mean() - points[:,1].std() * j)
        return indices
    
    def _get_convexhull(self, inliers, bound_point):
        hull = ConvexHull(inliers)
        for simplex in hull.simplices: 
            bound_point.append(simplex) 
        return bound_point
    
    def get_boundary_pt_indices(self, principalCp):
        cols = list(np.arange(principalCp.shape[1]))
        bound_point = []
        biglst = []
        combolst = []
        
        for a,b in itertools.combinations(cols,2):
            if a != b:
        #             plt.title(str(a+1)+" vs "+str(b+1))
        #             plt.scatter(principalCp[:,a],principalCp[:,b]); #plt.show();
                combo = np.vstack((principalCp[:,a],principalCp[:,b]))
                combo = np.transpose(combo)
                biglst.append(combo)
                combolst.append([a+1,b+1])
       
        for i in range(len(biglst)):
            points = np.array(biglst[i])
            # W/o outliers to be implemented
            if self.hv_rung_enabled: 
                for j in range(1,self.hv_stdevs + 1):
                    indices = self._points_indices(points, j)
                    inliers = points[~indices]
                    try:
                        bound_point = self._get_convexhull(inliers, bound_point)
                    except:
                        print('Componets:',combolst[i],'-> Unable to create convexHull at stdev %i. Skipping.'%j)

            else:
                indices = self._points_indices(points)
                inliers = points[~indices] 
                try:
                    bound_point = self._get_convexhull(inliers, bound_point)
                except Exception as ex:
                    print('Componets:',combolst[i],'Unable to create convexHull. Skipping.')
            
            if self.hv_collect_outliers:# With outliers
                indices = self._points_indices(points)
                outliers = points[indices]
                try:
                    bound_point = self._get_convexhull(outliers, bound_point)
                except Exception as ex:
                    print('Componets:',combolst[i],'Unable to create outlier convexHull. Skipping.')
        bd = np.concatenate(bound_point).ravel().tolist()
        indices_bounds=np.unique(bd)
        return indices_bounds #[:]
    
    def create_new_buf(self,indices_bounds,mydarray,bufferER):
        new_buf = dict() #this is the new HV buffer
        for i in indices_bounds:
            arr = mydarray[i]    
            keylist = list(bufferER.keys())
            if len(new_buf) == 0:
                count = 0
                for j in np.arange(len(keylist)): 
                    new_buf[keylist[j]] = []  
                    new_buf[keylist[j]].append(arr[count:count+bufferER[keylist[j]].shape[1]])
                    count += bufferER[keylist[j]].shape[1]
            else:
                count = 0
                for j in np.arange(len(keylist)):
                    new_buf[keylist[j]].append(arr[count:count+bufferER[keylist[j]].shape[1]])
                    count += bufferER[keylist[j]].shape[1]

        for key in new_buf.keys():
           new_buf[key] = np.array(new_buf[key])
        return new_buf
    
    def update_buffer(self,buffer):
        start_time = time.time()
        bufferER = self.clean_buffer(buffer) #HARDCODED FUNCTION
        clean_hv_buffer = self.clean_buffer(self.buffers)
        # 3 - Mix HV + ER Buffer
        if self.get_current_size()==0:
            buffer_d = bufferER
        else:
            ds = [bufferER,clean_hv_buffer]
            buffer_d = {}
            for k in bufferER:
                buffer_d[k] = np.concatenate(list(buffer_d[k] for buffer_d in ds))

        mydarray = np.concatenate([v for k,v in buffer_d.items()],1) #HARDCODED! 
        #May want to change later to take in some param in cmd line which is the extractable features u want to dimensionally reduce
        darray = np.concatenate([buffer_d[k] for k in ['o','u','ag']],1) 
        scaled = self.standardize_data(darray)
        principalCp = self.run_PCA(scaled)
        indices_bounds = self.get_boundary_pt_indices(principalCp)
        new_buf = self.create_new_buf(indices_bounds,mydarray,bufferER) #FUNCTION IS REALLYY HARDCODED!
        self.store_buffer(new_buf)
        print('Total julie time:',time.time()-start_time)

    def get_current_size(self):
        #print('[hv_buffer] -> get_current_size() invoked!')
        with self.lock:
            return self.current_size #* self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
