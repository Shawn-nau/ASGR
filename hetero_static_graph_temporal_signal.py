import torch
import numpy as np
from typing import List, Union,Tuple
from torch_geometric.data import HeteroData
import pandas as pd

Edge_Index = List[Union[np.ndarray, None]]
Edge_Weight = List[Union[np.ndarray, None]]
NNets = Union[int, None]
Node_Features_Timevarying = List[Union[np.ndarray, None]]
Node_Features_Static = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Target_features = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


class HeteroStaticGraphTemporalSignal(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        nNets: NNets,
        dynamic_features: Node_Features_Timevarying,
        static_features: Node_Features_Static,
        targets: Targets,
        targets_mask: Targets,
        target_features:Target_features,
        scales:List[Union[np.ndarray, None]],
        tdx:List[Union[np.ndarray, None]],
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.nNets = nNets
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.targets = targets
        self.targets_mask = targets_mask
        self.target_features = target_features
        self.additional_feature_keys = []
        self.scales = scales
        self.tdx = tdx
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.dynamic_features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.dynamic_features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return [torch.LongTensor(self.edge_index[i]) for i in range(self.nNets)]

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return [torch.FloatTensor(self.edge_weight[i]) for i in range(self.nNets)]

    def _get_features(self, time_index: int):
        if self.dynamic_features[time_index] is None:
            return self.dynamic_features[time_index]
        else:
            return torch.FloatTensor(self.dynamic_features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_target_mask(self, time_index: int):
        if self.targets_mask[time_index] is None:
            return self.targets_mask[time_index]
        else:
            if self.targets_mask[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets_mask[time_index])
            elif self.targets_mask[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets_mask[time_index])

    def _get_target_features(self, time_index: int):
        if self.target_features[time_index] is None:
            return self.target_features[time_index]
        else:
            return torch.FloatTensor(self.target_features[time_index])

    def _get_target_scale(self, time_index: int):
        if self.scales[time_index] is None:
            return self.scales[time_index]
        else:
            return torch.FloatTensor(self.scales[time_index])

    def _get_target_tdx(self, time_index: int):
        if self.tdx[time_index] is None:
            return self.tdx[time_index]
        else:
            return torch.FloatTensor(self.tdx[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __get_item__(self, time_index: int):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(time_index)
        mask = self._get_target_mask(time_index)
        z = self._get_target_features(time_index)
        scale = self._get_target_scale(time_index)
        
        additional_features = self._get_additional_features(time_index)
        snapshot = HeteroData()
        snapshot['node'].x = x
        snapshot['node'].y = y
        snapshot['node'].mask = mask
        snapshot['node'].z = z
        snapshot['node'].scale = scale
        snapshot['tdx'] = self.tdx[time_index]
        
        for i in range(self.nNets):
            snapshot['node','has_same_attr{}_with'.format(i),'node'].edge_index = edge_index[i]
            if edge_weight is None:
                snapshot['node','has_same_attr{}_with'.format(i),'node'].edge_attr = None
            else:
                napshot['node','has_same_attr{}_with'.format(i),'node'].edge_attr = edge_weight[i]
        
        if self.static_features is not None:
            snapshot['node'].f = torch.FloatTensor(self.static_features)
        else:
            snapshot['node'].f = None
        return snapshot

    def __next__(self):
        if self.t < len(self.dynamic_features):
            snapshot = self.__get_item__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self




class GCNdata:
    """
    transform pandas data to gcn data

    """

    def __init__(self,X,Y,mask,static_features = None, edges = None,edge_weight=None,):
        self._X = X   ## list of features(array) with nodes *time, 
        self._Y = Y  ## nodes * time, array
        self._edges = edges  ## list of nets(array) with 2*edges
        self._edge_weight = edge_weight  ## list of nets(array) with 2*edges
        self._nNets = len(self._edges)
        self._mask = mask  ## same shape with Y
        self._static_features = static_features

        self.eps = 1e-8


    def _get_targets_and_features(self):

        self.nfeatures = len(self._X)
        stacked_features_x = np.concatenate([self._X],axis =0 )
        stacked_features = np.concatenate([[self._Y],stacked_features_x],axis =0)

        self.features = [
            stacked_features[:,:,i : i + self.lags].swapaxes(0,1).copy()
            for i in range(self._Y.shape[1] - self.lags - self.h+1)
        ]
        self.targets = [
            self._Y[:,(i + self.lags):(i + self.lags + self.h)]
            for i in range(self._Y.shape[1] - self.lags - self.h+1)
        ]

        self.target_features = [
            stacked_features_x[:,:,(i + self.lags):(i + self.lags + self.h)].swapaxes(0,1)
            for i in range(self._Y.shape[1] - self.lags - self.h+1)
        ]
        
        self.targets_mask = [
            self._mask[:,(i + self.lags):(i + self.lags + self.h)]
            for i in range(self._Y.shape[1] - self.lags - self.h+1)
        ]

        self.tdx = [
            i + self.lags
            for i in range(self._Y.shape[1] - self.lags - self.h+1)
        ]

    def _y_normalize(self):
        ## only for postive forecasting
        scales = []
        
        for i in range(len(self.features )):
            
            scale_sum = np.sum(self.features[i][:,0,:]*(self.features[i][:,0,:]>0),axis=1) ## avoid sales of -1
            scale_count = np.sum((self.features[i][:,0,:]>0),axis=1)    
            scale = scale_sum/(scale_count+self.eps)        
            scale_mask = np.where(scale_count<4, 0,scale) ## mask new skus

            for t in range(self.features[i].shape[2]):
                self.features[i][:,0,t] = np.where(self.features[i][:,0,t]<=0, self.features[i][:,0,t],self.features[i][:,0,t]/(scale+self.eps))
            scales.append(scale_mask)

        self.scales = scales
    
    def get_dataset(self, lags: int = 4,h : int = 4) -> HeteroStaticGraphTemporalSignal:

        self.lags = lags
        self.h = h
        self._get_targets_and_features()
        self._y_normalize()
        dataset = HeteroStaticGraphTemporalSignal(
            self._edges, self._edge_weight, self._nNets,self.features,self._static_features, self.targets,self.targets_mask,self.target_features,self.scales,self.tdx,

        )
        return dataset
    

class SalesDatasetLoader:
    
    def __init__(self,data_file,node_file, net_file):   
        
        self._data = pd.read_csv(data_file)
        
        if node_file is not None:
            self._nodes = pd.read_csv(node_file)
        else:
            self._nodes = []
        #if np.min(self._nodes.values) > 0:
         #   self._nodes = self._nodes -1

        if net_file is not None:
            self.net = pd.read_csv(net_file).values.T   
            if np.min(self.net) > 0:
                self.net = self.net -1
            self.net = [self.net]          
        else:
            self.net = []
        
        
        self.special_days = [
            "Halloween",
            "Thanksgiving",
            "Christmas",
            "NewYear",
            "President",
            "Easter",
            "Memorial",
            "Labour",
            "fourthJuly",
        ]
            
           
        self.UPCs = pd.Series(range(1,self._data['UPCnumber'].max()+1),name='UPCnumber')
        self.stores = self._data.IRI_KEY.unique()
        self.features = ['price','PR','F','D']
        self.index = ["IRI_KEY","UPCnumber","WEEK"]
        
        
        if node_file is not None:
            self.attributes = self._nodes.columns.to_list()
        else:
            self.attributes = []
       
        
        
        self.target = 'UNITS'
        self.time_index = 'WEEK'
        self.store_index = "IRI_KEY"
        self.SKU_index = "UPCnumber"
        self._data.price = -np.log(self._data.price)
        
        
        self._data = self._time_align() # due to some stores may not open at some days
        self.gcndata = self._data_for_gcn()           
        
        
    def _time_align(self):
        
        df = self._data.set_index( self.index )
        
        Y = df[self.target].unstack()
        mask = Y.applymap(lambda x: 0 if np.isnan(x) else 1 )       

        Temp = []
        for X in self.features:
            temp = df[X].unstack().fillna(0)            
            Temp.append(temp.stack().rename(X))

        Temp.append(Y.fillna(0).stack().rename( self.target ))
        Temp.append(mask.stack().rename('MASK'))
        Temp = pd.concat(Temp,axis=1)

        data_special_days = df[self.special_days].groupby(level= self.time_index).agg('mean')
        Temp = Temp.reset_index().set_index(self.time_index).join(data_special_days).reset_index()
        Temp[self.store_index]= Temp[self.store_index].astype('category')
        return( Temp )
        
    def _data_for_gcn(self):     
        df = self._data
        if len(self._nodes)>0:
            attributes = self._nodes.values
        else:
            attributes = None
        graph_data = []
        for store in self.stores:
            Y = pd.merge(self.UPCs,df[df[self.store_index]== store].pivot(self.SKU_index,self.time_index,self.target),left_index=True,right_index=True,how='left').fillna(0).drop(self.SKU_index,axis=1).values
            MASK = pd.merge(self.UPCs,df[df[self.store_index]== store].pivot(self.SKU_index,self.time_index,'MASK'),left_index=True,right_index=True,how='left').fillna(0).drop(self.SKU_index,axis=1).values
            
            X = []
            
            for x in self.features + self.special_days:
                X.append(pd.merge(self.UPCs,df[df.IRI_KEY==store].pivot(self.SKU_index,self.time_index,x),left_index=True,right_index=True,how='left').fillna(0).drop(self.SKU_index,axis=1).values)
                       
            graph_data.append(GCNdata(X = X,Y = Y, mask = MASK, static_features = attributes, edges = self.net,edge_weight=None,))
        return(graph_data)
        


def temporal_signal_split(
    data_iterator, train_ratio: float = 0.8, H : float = 1
) -> Tuple[HeteroStaticGraphTemporalSignal, HeteroStaticGraphTemporalSignal]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """

    train_snapshots = int(train_ratio * data_iterator.snapshot_count)

    train_iterator = HeteroStaticGraphTemporalSignal(
        data_iterator.edge_index,
        data_iterator.edge_weight,
        data_iterator.nNets,
        data_iterator.dynamic_features[0:(train_snapshots-H+1)],
        data_iterator.static_features,
        
        data_iterator.targets[0:(train_snapshots-H+1)],
        data_iterator.targets_mask[0:(train_snapshots-H+1)],
        data_iterator.target_features[0:(train_snapshots-H+1)],
        
        data_iterator.scales[0:(train_snapshots-H+1)],
        data_iterator.tdx[0:(train_snapshots-H+1)],
        
    )

    test_iterator = HeteroStaticGraphTemporalSignal(
        data_iterator.edge_index,
        data_iterator.edge_weight,
        data_iterator.nNets,
        data_iterator.dynamic_features[train_snapshots:],
        data_iterator.static_features,
        
        data_iterator.targets[train_snapshots:],
        data_iterator.targets_mask[train_snapshots:],
        data_iterator.target_features[train_snapshots:],
        
        data_iterator.scales[train_snapshots:],
        data_iterator.tdx[train_snapshots:],
        
    )

    return train_iterator, test_iterator


