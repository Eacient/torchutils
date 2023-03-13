
import numpy as np

"""main usage
colorize_score: input cam, output heat-map or seg-map
colorize_label: input label, output seg-map
"""


def colorize_score(score_map, exclude_zero=False, normalize=True, by_hue=False):
    """
    return a colored map of cam-like score_map
    input:
        score_map: [C, H, W]
        by_hue: 
            if True, return heatmap-like fig[H,W,3]
            if not specified, return segmap-like fig[H,W,3], only support voc-style
        normalize:
            default True, return range in [0,1]
            if False, return range in unknow
        exclude_zero: only used when by_hue=False, ignore all-black color in voc clors
    output:
        [h, w, 3], heatmap or segmap
    """
    import matplotlib.colors
    if by_hue:
        aranged = np.arange(score_map.shape[0]) / (score_map.shape[0]) # [C]
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1) #[C, 3]
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color) #[C, 3]

        test = rgb_color[np.argmax(score_map, axis=0)] #[3]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test #[H, W, 1] * [3] -> [H, W, 3]

        if normalize:
            return test / (np.max(test) + 1e-5) # range to [0,1]
        else:
            return test

    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                     (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                     (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]

        test = VOC_color[np.argmax(score_map, axis=0)%22]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test
        if normalize:
            test /= np.max(test) + 1e-5

        return test


def colorize_displacement(disp):
    #[2, n]
    import matplotlib.colors
    import math

    a = (np.arctan2(-disp[0], -disp[1]) / math.pi + 1) / 2 # [n] divide and rerange to [0,1]

    r = np.sqrt(disp[0] ** 2 + disp[1] ** 2) # [n] radius
    s = r / np.max(r) #rerange to [0,1]
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1) #[n, 3]
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color) #[n, 3]

    return rgb_color


def colorize_label(label_map, normalize=True, by_hue=True, exclude_zero=False, outline=False):
    # input label_map, return colored_map
    label_map = label_map.astype(np.uint8) #[H, W]

    if by_hue:
        import matplotlib.colors
        sz = np.max(label_map) #[1]
        aranged = np.arange(sz) / sz #[C-1]
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1) #[C-1,3]
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color) #[C-1, 3]
        rgb_color = np.concatenate([np.zeros((1, 3)), rgb_color], axis=0) #[C, 3]

        test = rgb_color[label_map] #[H, W, 3]
    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                              (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]
        test = VOC_color[label_map]
        if normalize:
            test /= np.max(test) #[H, W, 3]

    if outline:
        edge = np.greater(np.sum(np.abs(test[:-1, :-1] - test[1:, :-1]), axis=-1) 
            + np.sum(np.abs(test[:-1, :-1] - test[:-1, 1:]), axis=-1), 0) #[H-1,W-1] 两种逐pos递减，只要一个方向有相差就为True，否则为False
        edge1 = np.pad(edge, ((0, 1), (0, 1)), mode='constant', constant_values=0) #[h,w]
        edge2 = np.pad(edge, ((1, 0), (1, 0)), mode='constant', constant_values=0) #[h,w]
        edge = np.repeat(np.expand_dims(np.maximum(edge1, edge2), -1), 3, axis=-1) #[h,w,3]

        test = np.maximum(test, edge)
    return 