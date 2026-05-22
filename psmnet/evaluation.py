import glob
import cv2
import numpy as np

def compute_epe(est_path, gt_path, min_disp, max_disp):
    """
    计算单张图的EPE（end-point-error）
    est_path: 预测视差图路径
    gt_path:  真值视差图路径
    min_disp, max_disp: 只评估在[min_disp, max_disp)范围内的像素
    """
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)  # 预测视差
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)    # 真值视差

    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)     # gt < max_disp
    mask2 = np.where(gt < min_disp, zeros, ones)      # gt >= min_disp
    mask = mask1 & mask2                              # 合并mask：有效像素

    error = np.sum(np.abs(est - gt) * mask)           # 绝对误差求和（只统计有效像素）
    nums = np.sum(mask)                               # 有效像素数
    epe = error / nums                                # 平均EPE

    return error, nums, epe

def compute_d1(est_path, gt_path, min_disp, max_disp):
    """
    计算单张图的D1错误率（3 pixel error率）
    """
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)
    mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask1 & mask2

    err_map = np.abs(est - gt) * mask             # 绝对误差图
    err_mask = err_map > 3                        # 大于3的为错误像素
    err_disps = np.sum(err_mask.astype('float32'))# 错误像素数
    nums = np.sum(mask)                           # 有效像素数
    d1 = err_disps / nums                         # D1错误率

    return err_disps, nums, d1

def evaluate(est_path, gt_path, min_disp, max_disp):
    """
    单张图片评估并打印EPE/D1
    """
    error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
    print('Sum of absolute error: %f, num of valid pixels: %d, end-point-error: %f'
          % (error, int(nums), epe))
    err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
    print('Num of error disparities: %d, num of valid pixels: %d, d1: %f'
          % (int(err_disps), int(nums), d1))

def evaluate_all(est_dir, gt_dir, min_disp, max_disp):
    """
    批量评估文件夹下所有预测/真值对，打印整体EPE和D1
    """
    est_paths = glob.glob(est_dir + '/*')   # 预测结果文件列表
    gt_paths = glob.glob(gt_dir + '/*')     # 真值文件列表
    est_paths.sort()
    gt_paths.sort()
    assert len(est_paths) == len(gt_paths)

    # 统计整体EPE
    total_error, total_nums = 0, 0
    for est_path, gt_path in zip(est_paths, gt_paths):
        error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
        total_error += error
        total_nums += nums
    print('\nEnd-point-error: %f pixel' % (total_error / total_nums))

    # 统计整体D1
    total_err_disps, total_nums = 0, 0
    for est_path, gt_path in zip(est_paths, gt_paths):
        err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
        total_err_disps += err_disps
        total_nums += nums
    print('\nD1: %f' % (total_err_disps * 100 / total_nums), "%")  # 百分比形式