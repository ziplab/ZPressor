import torch


def center_filter(x, center_views, is_view_list=False):
    """
    Filters the input tensor x to only include the center views
    Args:
        x: input tensor, shape (batch_size, num_views, ...) or list of input tensors, shape [(batch_size, ...)](length=num_views)
        center_views: list of center views, sampled pointcloud index, [batch_size, num_center_views]
    """
    if center_views is None:
        return x
    if not is_view_list and isinstance(x, list):
        return [center_filter(x_i, center_views) for x_i in x]
    if is_view_list:
        assert isinstance(x, list)
        # convert list of tensors to tensor
        x = torch.stack(x, dim=1)
    
    batch_size = x.size(0)
    batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1)
    output_x = x[batch_idx, center_views]
    if is_view_list:
        output_x = output_x.unbind(1)
    return output_x
    

if __name__ == "__main__":
    x = torch.rand(2, 3, 1, 1, 1)
    center_views = torch.tensor([[0], [1]])
    print(x)
    print(center_filter(x, center_views))

    x_list = [torch.rand(2, 1, 1, 1, 1), torch.rand(2, 1, 1, 1, 1)]
    print(x_list)
    print(center_filter(x_list, center_views))