"""Test CNN modules."""
import torch
from torch import rand

from audlib.nn.strfnet import STRFLayerFC, STRFNetWASPAA2019 as STRFNet


def test_strflayer():
    """Test STRFLayerFC."""
    layer = STRFLayerFC(100, 24, 1, 1.5, 16, (100, 100), (4, 4), 5, 60, 32)
    assert layer(rand(1, 100, 100)).detach().numpy().shape == (1, 32, 4, 4)
    layer = STRFLayerFC(100, 24, 1, 1.5, 16, (500, 100), (4, 4), 5, 60, 32)
    assert layer(rand(1, 500, 100)).detach().numpy().shape == (1, 32, 4, 4)
    return


def test_strfnet():
    """Test STRFNetWASPAA2019."""
    # TODO: Add test case.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fr = 100
    bpo = 24
    suptimes = [.5]
    supocts = [2]
    nkerns = [16]
    num_classes = 2
    indim = (100, 100)
    mlp_hidden = [512, 128, 32]
    net = STRFNet(fr, bpo, suptimes, supocts, nkerns,
                  num_classes, indim, mlp_hidden).to(device)
    print(net)
    res = net(torch.rand(32, 100, 100).to(device))  # simulation
    print(res.shape)
    loss = res.sum()
    loss.backward()

    # Grab STRF and visualize
    strfs = net.strflayers[0].strfconv.strfgen().cpu().detach().numpy()
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(142)
                     nrows_ncols=(4, 4),
                     axes_pad=0.0,
                     share_all=True,
                     label_mode="L",
                     cbar_location="top",
                     cbar_mode="single",
                     )
    for i in range(16):
        im = grid[i].pcolormesh(strfs[i])
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

    plt.show()
    """
    print(strfs.shape)


if __name__ == "__main__":
    test_strflayer()
    test_strfnet()
