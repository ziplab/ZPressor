import torch
from torch import nn
from einops import rearrange
from .attention import CrossAttention, Attention

from sklearn.cluster import KMeans

class ZPressor(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=1, num_layers=6, no_self_attn=False):
        super(ZPressor, self).__init__()
        self.num_heads = num_heads
        self.no_self_attn = no_self_attn
        self.num_layers = num_layers

        # If embed_dim is a number, set num_heads=1
        if isinstance(embed_dim, int):
            self.feature_layer_num = 1
            self.embed_dim = [embed_dim]
        else:
            self.feature_layer_num = len(embed_dim)
            self.embed_dim = embed_dim

        cross_ffn = True
        if not self.no_self_attn:
            self.self_attn_layers = nn.ModuleList([
                nn.ModuleList([
                    Attention(
                        dim=self.embed_dim[i],
                        dim_head=self.embed_dim[i] // num_heads,
                        heads=num_heads,
                        ffn=True
                    ) for _ in range(num_layers)
                ]) for i in range(self.feature_layer_num)
            ])
            cross_ffn = False

        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttention(
                    dim=self.embed_dim[i],
                    num_heads=num_heads,
                    ffn=cross_ffn
                ) for _ in range(num_layers)
            ]) for i in range(self.feature_layer_num)
        ])


    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """

        device = xyz.device
        B, N, C = xyz.shape

        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10

        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        barycenter = torch.sum((xyz), 1)
        barycenter = barycenter / xyz.shape[1]
        barycenter = barycenter.view(B, 1, 3)

        dist = torch.sum((xyz - barycenter) ** 2, -1)
        farthest = torch.max(dist, 1)[1]

        selected_mask = torch.zeros(B, N, dtype=torch.bool).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest
            selected_mask[batch_indices, farthest] = 1

            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            dist[selected_mask] = float(0)

            mask = dist < distance
            distance[mask] = dist[mask]

            farthest = torch.max(distance, -1)[1]

        return centroids
    
    def assign_points_to_clusters(self, xyz, centroids):
        """
        Divide point cloud into npoint clusters, assigning each point to the nearest centroid.

        Input:
            xyz: pointcloud data, [B, N, 3]
            centroids: sampled pointcloud index, [B, npoint]

        Return:
            clusters: cluster index for each point, shape [B, N]
        """
        device = xyz.device
        B, N, C = xyz.shape
        npoint = centroids.shape[1]

        # Calculate distance from each point to each centroid
        clusters = torch.zeros(B, N, dtype=torch.long).to(device)

        for b in range(B):
            # Get point cloud xyz and centroids for current batch
            points = xyz[b]  # Shape [N, 3]
            centroids_b = centroids[b]  # Shape [npoint]
            
            # Calculate Euclidean distance from each point to each centroid
            dist = torch.sum((points.unsqueeze(1) - points[centroids_b])**2, dim=-1)  # Shape [N, npoint]
            
            # Assign each point to the nearest centroid
            clusters[b] = torch.argmin(dist, dim=1)  # Shape [N]
        
        return clusters

    def kmeans_cluster_view_positions(self, view_positions, cluster_num):
        """
        Cluster view positions using K-means
        
        Input:
            view_positions: view position data, [B, N, 3]
            cluster_num: number of clusters
            
        Return:
            centroids: cluster center indices, [B, cluster_num]
            clusters: cluster assignments for each point, [B, N]
        """
        device = view_positions.device
        B, N, C = view_positions.shape
        
        centroids = torch.zeros(B, cluster_num, dtype=torch.long).to(device)
        clusters = torch.zeros(B, N, dtype=torch.long).to(device)
        
        for b in range(B):
            # Cluster view positions for current batch using K-means
            kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(view_positions[b].detach().cpu().numpy())
            clusters[b] = torch.tensor(cluster_labels).to(device)
            
            # Find the point closest to cluster center as centroid for each cluster
            for j in range(cluster_num):
                cluster_indices = torch.where(clusters[b] == j)[0]
                if len(cluster_indices) > 0:
                    cluster_points = view_positions[b][cluster_indices]
                    center_kmeans = torch.tensor(kmeans.cluster_centers_[j]).to(device)
                    distances = torch.norm(cluster_points - center_kmeans, dim=1)
                    closest_idx = cluster_indices[torch.argmin(distances)]
                    centroids[b, j] = closest_idx
                else:
                    # If a cluster is empty, select the first point as center
                    centroids[b, j] = 0
        
        return centroids, clusters

    def clust_token_kmeans(self, x):
        b, v, f = x.shape
        output = []
        centers = []
        
        for i in range(b):
            kmeans = KMeans(n_clusters=self.cluster_num, random_state=42, n_init=10)
            y = kmeans.fit_predict(x[i].detach().cpu().numpy())
            output.append(torch.tensor(y).cuda())
            y_tensor = torch.tensor(y).cuda()
            
            center = []
            for j in range(self.cluster_num):
                cluster_indices = torch.where(y_tensor == j)[0]
                if len(cluster_indices) > 0:
                    cluster_x = x[i][cluster_indices]
                    center_kmeans = torch.tensor(kmeans.cluster_centers_[j]).cuda()
                    distance = torch.norm(cluster_x - center_kmeans, dim=1)
                    closest = cluster_indices[torch.argmin(distance)]
                    center.append(closest)
                else:
                    center.append(torch.tensor(0).cuda())
            
            center = torch.stack(center)
            centers.append(center)
            
        centers = torch.stack(centers)
        output = torch.stack(output, dim=0)
        return output, centers
    
    def cross_matching(self, x, centroids, clusters, layer_idx, pre_output=None, feature_layer_idx=0):
        # x: (b, v, n, f)
        output = []
        center_feats = None
        if pre_output is None:
            use_output = False
        else:
            use_output = True
        B, V, _, _ = x.shape
        for i in range(B):
            cls_output = []
            for j in range(self.cluster_num):
                center_idx = centroids[i, j]
                if use_output:
                    center_x = pre_output[i][j].unsqueeze(0)
                else:
                    center_x = x[i, center_idx].unsqueeze(0)
                idx = clusters[i] == j
                idx[center_idx] = False
                # Concatenate all rows of x[i][idx] (b, n, d) -> (bn, d)
                cls_x = x[i][idx].reshape(-1, x.shape[-1]).unsqueeze(0)
                if cls_x.shape[1] > 0:
                    center_x = self.cross_attn_layers[feature_layer_idx][layer_idx](center_x, cls_x)
                cls_output.append(center_x)
            cls_output = torch.cat(cls_output, dim=0)
            output.append(cls_output)
        output = torch.stack(output, dim=0)
        return output, center_feats
    
    
    def self_matching(self, x, layer_idx, feature_layer_idx):
        b, v, n, f = x.shape
        output = []
        for i in range(b):
            output.append(self.self_attn_layers[feature_layer_idx][layer_idx](x[i]))
        output = torch.stack(output, dim=0)
        return output
    
    def forward(self, x, extrinsics=None, cls_token=None, cluster_num=8):
        self.cluster_num = cluster_num
        # assert (extrinsics is not None) ^ (cls_token is not None)

        if not isinstance(x, list):
            x = [x]
        
        if cls_token is None:
            candidate_view_positions = extrinsics[:, :, :3, -1]
            centroids = self.farthest_point_sample(candidate_view_positions, self.cluster_num)
            clusters = self.assign_points_to_clusters(candidate_view_positions, centroids)
        else:
            clusters, centroids = self.clust_token_kmeans(cls_token)

        results = []
        for i in range(self.feature_layer_num):
            b, v, f, h, w = x[i].shape

            x[i] = rearrange(x[i], 'b v f h w -> b v (h w) f', b=b, v=v)
            x_original = x[i].clone()

            output = None
            for layer_idx in range(self.num_layers):
                output, center_feats = self.cross_matching(x[i], centroids, clusters, layer_idx, output, i)
                if not self.no_self_attn:
                    output = self.self_matching(output, layer_idx, i)
                    
            output = rearrange(output, 'b v (h w) f -> b v f h w', b=b, v=self.cluster_num, h=h, w=w)
            results.append(output)

        if self.feature_layer_num > 1:
            return results, centroids
        else:
            return results[0], centroids


def simple_test():
    model = ZPressor().cuda()
    x = torch.randn(2, 5, 1024, 14, 14).cuda()
    cls_token = torch.randn(2, 5, 1024).cuda()
    extrinsics = torch.randn(2, 5, 4, 4).cuda()
    output, centroids = model(x, extrinsics=extrinsics)
    print(output.shape)
    print(centroids)
    output, centroids = model(x, cls_token=cls_token)
    print(output.shape)
    print(centroids)

if __name__ == '__main__':
    simple_test()
