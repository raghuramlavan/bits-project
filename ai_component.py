from numpy import random
from scipy.spatial import distance,cKDTree


def do_kdtree(next_centroids,current_points):
    mytree = cKDTree(next_centroids)
    dist, indexes = mytree.query(current_points)
    return dist,indexes

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def extract_features(feat_buff_centroid,feat_buff_values,feat_buff_size,feat_buff_ptr):

    feature = [[0 for i in range(feat_buff_size)] for i in range(len(feat_buff_centroid[feat_buff_ptr]))]
    next_frames = feat_buff_centroid[(feat_buff_ptr+1)%feat_buff_size]
    
    now_cent = feat_buff_centroid[feat_buff_ptr]
    
    dist,indexes = do_kdtree(next_frames,now_cent)

    for deapth in range(2,feat_buff_size+1):
        
        now_cent = next_frames[indexes]
        next_frames = feat_buff_centroid[(feat_buff_ptr+deapth)%feat_buff_size]
        feat_buff_ptr = (feat_buff_ptr + 1)% feat_buff_size
        dist,indexes=do_kdtree(next_frames,now_cent)
        #print(f"{next_frames.shape},{feat_buff_centroid[0].shape} {dist.shape} - len- {len(indexes)}")
        for ii in range(len(dist)):
            #print(f"now_cem -> {now_cent.shape}  - dist -> {len(dist)} - ii -> {ii} deapth -> {deapth}")
            #alignment is eveythong
  
            feature[ii][deapth-2]=dist[ii]
  
    return feature

    