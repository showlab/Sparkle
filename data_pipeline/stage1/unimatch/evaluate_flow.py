from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloader.flow.datasets import FlyingChairs, FlyingThings3D, MpiSintel, KITTI
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile, flow_to_image
import imageio

from utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from unimatch.geometry import forward_backward_consistency_check
from utils.file_io import extract_video
from tqdm import tqdm
import json
import cv2

from base_dataset import BaseVideoDataset


@torch.no_grad()
def create_sintel_submission(
    model,
    output_path="sintel_submission",
    padding_factor=8,
    save_vis_flow=False,
    no_save_flo=False,
    attn_type="swin",
    attn_splits_list=False,
    corr_radius_list=False,
    prop_radius_list=False,
    num_reg_refine=1,
    inference_size=None,
):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    for dstype in ["clean", "final"]:
        test_dataset = MpiSintel(split="test", aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if inference_size is not None:
                assert isinstance(inference_size, list) or isinstance(
                    inference_size, tuple
                )
                ori_size = image1.shape[-2:]
                image1 = F.interpolate(
                    image1, size=inference_size, mode="bilinear", align_corners=True
                )
                image2 = F.interpolate(
                    image2, size=inference_size, mode="bilinear", align_corners=True
                )
            else:
                padder = InputPadder(image1.shape, padding_factor=padding_factor)
                image1, image2 = padder.pad(image1, image2)

            results_dict = model(
                image1,
                image2,
                attn_type=attn_type,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                num_reg_refine=num_reg_refine,
                task="flow",
            )

            flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

            # resize back
            if inference_size is not None:
                flow_pr = F.interpolate(
                    flow_pr, size=ori_size, mode="bilinear", align_corners=True
                )
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()
            else:
                flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)

            if save_vis_flow:
                vis_flow_file = output_file.replace(".flo", ".png")
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(
    model,
    output_path="kitti_submission",
    padding_factor=8,
    save_vis_flow=False,
    attn_type="swin",
    attn_splits_list=False,
    corr_radius_list=False,
    prop_radius_list=False,
    num_reg_refine=1,
    inference_size=None,
):
    """Create submission for the KITTI leaderboard"""
    model.eval()
    test_dataset = KITTI(split="testing", aug_params=None)

    print("%d test samples" % len(test_dataset))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]

        image1 = image1[None].cuda()  # [1, 3, H, W]
        image2 = image2[None].cuda()

        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(
                image1, size=inference_size, mode="bilinear", align_corners=True
            )
            image2 = F.interpolate(
                image2, size=inference_size, mode="bilinear", align_corners=True
            )
        else:
            padder = InputPadder(
                image1.shape, mode="kitti", padding_factor=padding_factor
            )
            image1, image2 = padder.pad(image1, image2)

        results_dict = model(
            image1,
            image2,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            task="flow",
        )

        flow_pr = results_dict["flow_preds"][-1]

        # resize back
        if inference_size is not None:
            flow_pr = F.interpolate(
                flow_pr, size=ori_size, mode="bilinear", align_corners=True
            )
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()
        else:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(
    model,
    with_speed_metric=False,
    attn_type="swin",
    attn_splits_list=None,
    corr_radius_list=None,
    prop_radius_list=None,
    num_reg_refine=1,
):
    """Perform evaluation on the FlyingChairs (test) split"""
    model.eval()
    epe_list = []
    results = {}

    if with_speed_metric:
        s0_10_list = []
        s10_40_list = []
        s40plus_list = []

    val_dataset = FlyingChairs(split="validation")

    print("Number of validation image pairs: %d" % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        results_dict = model(
            image1,
            image2,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            task="flow",
        )

        flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if with_speed_metric:
            flow_gt_speed = torch.sum(flow_gt**2, dim=0).sqrt()
            valid_mask = flow_gt_speed < 10
            if valid_mask.max() > 0:
                s0_10_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
            if valid_mask.max() > 0:
                s10_40_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = flow_gt_speed > 40
            if valid_mask.max() > 0:
                s40plus_list.append(epe[valid_mask].cpu().numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)
    px3 = np.mean(epe_all > 3)
    px5 = np.mean(epe_all > 5)
    print(
        "Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f"
        % (epe, px1, px3, px5)
    )
    results["chairs_epe"] = epe
    results["chairs_1px"] = px1
    results["chairs_3px"] = px3
    results["chairs_5px"] = px5

    if with_speed_metric:
        s0_10 = np.mean(np.concatenate(s0_10_list))
        s10_40 = np.mean(np.concatenate(s10_40_list))
        s40plus = np.mean(np.concatenate(s40plus_list))

        print(
            "Validation Chairs s0_10: %.3f, s10_40: %.3f, s40+: %.3f"
            % (s0_10, s10_40, s40plus)
        )

        results["chairs_s0_10"] = s0_10
        results["chairs_s10_40"] = s10_40
        results["chairs_s40+"] = s40plus

    return results


@torch.no_grad()
def validate_things(
    model,
    padding_factor=8,
    with_speed_metric=False,
    max_val_flow=400,
    val_things_clean_only=True,
    attn_type="swin",
    attn_splits_list=False,
    corr_radius_list=False,
    prop_radius_list=False,
    num_reg_refine=1,
):
    """Peform validation using the Things (test) split"""
    model.eval()
    results = {}

    for dstype in ["frames_cleanpass", "frames_finalpass"]:
        if val_things_clean_only:
            if dstype == "frames_finalpass":
                continue

        val_dataset = FlyingThings3D(
            dstype=dstype,
            test_set=True,
            validate_subset=True,
        )
        print("Number of validation image pairs: %d" % len(val_dataset))
        epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            results_dict = model(
                image1,
                image2,
                attn_type=attn_type,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                num_reg_refine=num_reg_refine,
                task="flow",
            )
            flow_pr = results_dict["flow_preds"][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt**2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

            if with_speed_metric:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (
                    (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                )
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_list = np.mean(np.concatenate(epe_list))

        epe = np.mean(epe_list)

        if dstype == "frames_cleanpass":
            dstype = "things_clean"
        if dstype == "frames_finalpass":
            dstype = "things_final"

        print("Validation Things test set (%s) EPE: %.3f" % (dstype, epe))
        results[dstype + "_epe"] = epe

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print(
                "Validation Things test (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f"
                % (dstype, s0_10, s10_40, s40plus)
            )

            results[dstype + "_s0_10"] = s0_10
            results[dstype + "_s10_40"] = s10_40
            results[dstype + "_s40+"] = s40plus

    return results


@torch.no_grad()
def validate_sintel(
    model,
    count_time=False,
    padding_factor=8,
    with_speed_metric=False,
    evaluate_matched_unmatched=False,
    attn_type="swin",
    attn_splits_list=False,
    corr_radius_list=False,
    prop_radius_list=False,
    num_reg_refine=1,
):
    """Peform validation using the Sintel (train) split"""
    model.eval()
    results = {}

    if count_time:
        total_time = 0
        num_runs = 100

    for dstype in ["clean", "final"]:
        val_dataset = MpiSintel(
            split="training",
            dstype=dstype,
            load_occlusion=evaluate_matched_unmatched,
        )

        print("Number of validation image pairs: %d" % len(val_dataset))
        epe_list = []

        if evaluate_matched_unmatched:
            matched_epe_list = []
            unmatched_epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            if evaluate_matched_unmatched:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(
                    flow_gt.unsqueeze(0)
                ).squeeze(
                    0
                )  # [H, W]

            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            if count_time and val_id >= 5:  # 5 warmup
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            results_dict = model(
                image1,
                image2,
                attn_type=attn_type,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                num_reg_refine=num_reg_refine,
                task="flow",
            )

            flow_pr = results_dict["flow_preds"][-1]

            if count_time and val_id >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if val_id >= num_runs + 4:
                    break

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if evaluate_matched_unmatched:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

            if with_speed_metric:
                flow_gt_speed = torch.sum(flow_gt**2, dim=0).sqrt()
                valid_mask = flow_gt_speed < 10
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = flow_gt_speed > 40
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)

        dstype_ori = dstype

        print(
            "Validation Sintel (%s) EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f"
            % (dstype_ori, epe, px1, px3, px5)
        )

        dstype = "sintel_" + dstype

        results[dstype + "_epe"] = np.mean(epe_list)
        results[dstype + "_1px"] = px1
        results[dstype + "_3px"] = px3
        results[dstype + "_5px"] = px5

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            print(
                "Validation Sintel (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f"
                % (dstype_ori, s0_10, s10_40, s40plus)
            )

            results[dstype + "_s0_10"] = s0_10
            results[dstype + "_s10_40"] = s10_40
            results[dstype + "_s40+"] = s40plus

        if count_time:
            print("Time: %.6fs" % (total_time / num_runs))
            break  # only the clean pass when counting time

        if evaluate_matched_unmatched:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))

            print(
                "Validatation Sintel (%s) matched epe: %.3f, unmatched epe: %.3f"
                % (dstype_ori, matched_epe, unmatched_epe)
            )

            results[dstype + "_matched"] = matched_epe
            results[dstype + "_unmatched"] = unmatched_epe

    return results


@torch.no_grad()
def validate_kitti(
    model,
    padding_factor=8,
    with_speed_metric=False,
    average_over_pixels=True,
    attn_type="swin",
    attn_splits_list=False,
    corr_radius_list=False,
    prop_radius_list=False,
    num_reg_refine=1,
    debug=False,
):
    """Peform validation using the KITTI-2015 (train) split"""
    model.eval()

    val_dataset = KITTI(split="training")
    print("Number of validation image pairs: %d" % len(val_dataset))

    out_list, epe_list = [], []
    results = {}

    if with_speed_metric:
        if average_over_pixels:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []
        else:
            s0_10_epe_sum = 0
            s0_10_valid_samples = 0
            s10_40_epe_sum = 0
            s10_40_valid_samples = 0
            s40plus_epe_sum = 0
            s40plus_valid_samples = 0

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode="kitti", padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        results_dict = model(
            image1,
            image2,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            task="flow",
        )

        # useful when using parallel branches
        flow_pr = results_dict["flow_preds"][-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        if with_speed_metric:
            # flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            flow_gt_speed = mag

            if average_over_pixels:
                valid_mask = (flow_gt_speed < 10) * (
                    valid_gt >= 0.5
                )  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (
                    (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                )
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

            else:
                valid_mask = (flow_gt_speed < 10) * (
                    valid_gt >= 0.5
                )  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s0_10_valid_samples += 1

                valid_mask = (
                    (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                )
                if valid_mask.max() > 0:
                    s10_40_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s10_40_valid_samples += 1

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s40plus_valid_samples += 1

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        if average_over_pixels:
            epe_list.append(epe[val].cpu().numpy())
        else:
            epe_list.append(epe[val].mean().item())

        out_list.append(out[val].cpu().numpy())

        if debug:
            if val_id > 10:
                break

    if average_over_pixels:
        epe_list = np.concatenate(epe_list)
    else:
        epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI EPE: %.3f, F1-all: %.3f" % (epe, f1))
    results["kitti_epe"] = epe
    results["kitti_f1"] = f1

    if with_speed_metric:
        if average_over_pixels:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))
        else:
            s0_10 = s0_10_epe_sum / s0_10_valid_samples
            s10_40 = s10_40_epe_sum / s10_40_valid_samples
            s40plus = s40plus_epe_sum / s40plus_valid_samples

        print(
            "Validation KITTI s0_10: %.3f, s10_40: %.3f, s40+: %.3f"
            % (s0_10, s10_40, s40plus)
        )

        results["kitti_s0_10"] = s0_10
        results["kitti_s10_40"] = s10_40
        results["kitti_s40+"] = s40plus

    return results


@torch.no_grad()
def inference_flow(
    model,
    inference_dir,
    inference_video=None,
    output_path="output",
    padding_factor=8,
    inference_size=None,
    save_flo_flow=False,  # save raw flow prediction as .flo
    attn_type="swin",
    attn_splits_list=None,
    corr_radius_list=None,
    prop_radius_list=None,
    num_reg_refine=1,
    pred_bidir_flow=False,
    pred_bwd_flow=False,
    fwd_bwd_consistency_check=False,
    save_video=False,
    concat_flow_img=False,
):
    """Inference on a directory or a video"""
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if save_video:
        assert inference_video is not None

    fixed_inference_size = inference_size
    transpose_img = False

    if inference_video is not None:
        filenames, fps = extract_video(inference_video)  # list of [H, W, 3]
    else:
        filenames = sorted(
            glob(inference_dir + "/*.png") + glob(inference_dir + "/*.jpg")
        )
    print("%d images found" % len(filenames))

    vis_flow_preds = []
    ori_imgs = []

    for test_id in range(0, len(filenames) - 1):
        if (test_id + 1) % 50 == 0:
            print("predicting %d/%d" % (test_id + 1, len(filenames)))

        if inference_video is not None:
            image1 = filenames[test_id]
            image2 = filenames[test_id + 1]
        else:
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        if concat_flow_img:
            ori_imgs.append(image1)

        image1 = (
            torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        )
        image2 = (
            torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        )

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [
            int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor,
        ]

        # resize to nearest size or specified size
        inference_size = (
            nearest_size if fixed_inference_size is None else fixed_inference_size
        )

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(
                image1, size=inference_size, mode="bilinear", align_corners=True
            )
            image2 = F.interpolate(
                image2, size=inference_size, mode="bilinear", align_corners=True
            )

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(
            image1,
            image2,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            task="flow",
            pred_bidir_flow=pred_bidir_flow,
        )

        flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(
                flow_pr, size=ori_size, mode="bilinear", align_corners=True
            )
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        if inference_video is not None:
            output_file = os.path.join(output_path, "%04d_flow.png" % test_id)
        else:
            output_file = os.path.join(
                output_path, os.path.basename(filenames[test_id])[:-4] + "_flow.png"
            )

        if inference_video is not None and save_video:
            vis_flow_preds.append(flow_to_image(flow))
        else:
            # save vis flow
            save_vis_flow_tofile(flow, output_file)

        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]
            flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            if inference_video is not None:
                output_file = os.path.join(output_path, "%04d_flow_bwd.png" % test_id)
            else:
                output_file = os.path.join(
                    output_path,
                    os.path.basename(filenames[test_id])[:-4] + "_flow_bwd.png",
                )

            # save vis flow
            save_vis_flow_tofile(flow_bwd, output_file)

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                fwd_occ, bwd_occ = forward_backward_consistency_check(
                    flow_pr[:1], flow_pr[1:]
                )  # [1, H, W] float

                if inference_video is not None:
                    fwd_occ_file = os.path.join(
                        output_path, "%04d_occ_fwd.png" % test_id
                    )
                    bwd_occ_file = os.path.join(
                        output_path, "%04d_occ_bwd.png" % test_id
                    )
                else:
                    fwd_occ_file = os.path.join(
                        output_path,
                        os.path.basename(filenames[test_id])[:-4] + "_occ_fwd.png",
                    )
                    bwd_occ_file = os.path.join(
                        output_path,
                        os.path.basename(filenames[test_id])[:-4] + "_occ_bwd.png",
                    )

                Image.fromarray(
                    (fwd_occ[0].cpu().numpy() * 255.0).astype(np.uint8)
                ).save(fwd_occ_file)
                Image.fromarray(
                    (bwd_occ[0].cpu().numpy() * 255.0).astype(np.uint8)
                ).save(bwd_occ_file)

        if save_flo_flow:
            if inference_video is not None:
                output_file = os.path.join(output_path, "%04d_pred.flo" % test_id)
            else:
                output_file = os.path.join(
                    output_path, os.path.basename(filenames[test_id])[:-4] + "_pred.flo"
                )
            frame_utils.writeFlow(output_file, flow)
            if pred_bidir_flow:
                if inference_video is not None:
                    output_file_bwd = os.path.join(
                        output_path, "%04d_pred_bwd.flo" % test_id
                    )
                else:
                    output_file_bwd = os.path.join(
                        output_path,
                        os.path.basename(filenames[test_id])[:-4] + "_pred_bwd.flo",
                    )
                frame_utils.writeFlow(output_file_bwd, flow_bwd)

    if save_video:
        suffix = "_flow_img.mp4" if concat_flow_img else "_flow.mp4"
        output_file = os.path.join(
            output_path, os.path.basename(inference_video)[:-4] + suffix
        )

        if concat_flow_img:
            results = []
            assert len(ori_imgs) == len(vis_flow_preds)

            concat_axis = 0 if ori_imgs[0].shape[0] < ori_imgs[0].shape[1] else 1
            for img, flow in zip(ori_imgs, vis_flow_preds):
                concat = np.concatenate((img, flow), axis=concat_axis)
                results.append(concat)
        else:
            results = vis_flow_preds

        imageio.mimwrite(output_file, results, fps=fps, quality=8)

    print("Done!")


def compute_dual_motion_features(flow_tensor):
    """
    计算光流的绝对运动强度和减去均值后的相对运动强度。
    假设 flow_tensor 形状为 (2, H, W)
    """
    # 如果你的张量形状是 (H, W, 2)，请解除下面这行的注释进行转换
    # flow_tensor = flow_tensor.permute(2, 0, 1)

    u = flow_tensor[0, :, :]
    v = flow_tensor[1, :, :]

    # ==========================================
    # 1. 计算绝对运动强度 (包含相机运镜)
    # ==========================================
    mag_abs = torch.sqrt(u**2 + v**2)
    mean_motion = torch.mean(mag_abs).item()

    # ==========================================
    # 2. 计算减去均值后的相对运动强度 (剔除全局平移)
    # ==========================================
    # 算出当前帧 X 和 Y 方向的全局平均位移 (代表背景/相机的移动)
    u_mean = torch.mean(u)
    v_mean = torch.mean(v)

    # 每个像素减去全局位移
    u_rel = u - u_mean
    v_rel = v - v_mean

    # 计算相对幅度
    mag_rel = torch.sqrt(u_rel**2 + v_rel**2)
    mean_relative_motion = torch.mean(mag_rel).item()

    return {
        "mean_motion": round(mean_motion, 4),
        "mean_relative_motion": round(mean_relative_motion, 4),
    }


def detect_camera_motion_from_flow(
    optical_flows, step=16, inlier_thresh=0.5, motion_thresh=1.0
):
    """
    通过光流序列判定每一帧是否存在相机移动。

    参数:
    - optical_flows: np.ndarray, 形状为 [T, 2, H, W]，Unimatch 输出的光流
    - step: int, 降采样步长。step=16 意味着每 16x16 的网格只取一个点进行拟合，极大提升速度
    - inlier_thresh: float, 判定为背景（统一运动）的内点比例阈值
    - motion_thresh: float, 判定为相机移动的最小平均像素位移

    返回:
    - camera_moving: list of bool, 长度为 T，标记每一帧相机是否在移动
    - motion_stats: dict, 包含每一帧的内点比例和相机位移量等统计信息的列表
    """
    T, C, H, W = optical_flows.shape
    assert C == 2, "光流通道数必须为 2 (u, v)"

    # 1. 提前构建全局的源坐标网格 (X, Y)
    # 使用 meshgrid 生成图像每个像素的坐标
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # 进行空间降采样，减少 RANSAC 计算量
    X_sub = X[::step, ::step].astype(np.float32)
    Y_sub = Y[::step, ::step].astype(np.float32)

    # 将二维网格展平为 (N, 1, 2) 形状，这是 OpenCV 函数要求的标准格式
    src_pts = np.stack([X_sub, Y_sub], axis=-1).reshape(-1, 1, 2)

    camera_moving = []
    motion_stats = {
        "inlier_ratio": [],
        "bg_motion": [],
    }

    # print(f"开始处理 {T} 帧光流数据，每帧采样点数: {src_pts.shape[0]} ...")

    for t in range(T):
        flow_u = optical_flows[t, 0, :, :]
        flow_v = optical_flows[t, 1, :, :]

        # 提取降采样后的光流向量
        u_sub = flow_u[::step, ::step].astype(np.float32)
        v_sub = flow_v[::step, ::step].astype(np.float32)

        # 2. 计算目标坐标：目标坐标 = 源坐标 + 光流位移
        dst_X = X_sub + u_sub
        dst_Y = Y_sub + v_sub
        dst_pts = np.stack([dst_X, dst_Y], axis=-1).reshape(-1, 1, 2)

        # 3. RANSAC 求解单应性矩阵 (Homography)
        # 允许最大 3.0 像素的重投影误差 (Reprojection Error)
        H_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H_matrix is None or mask is None:
            # 极端情况：没有任何结构化运动，或者画面完全混乱
            camera_moving.append(False)
            motion_stats["inlier_ratio"].append(0.0)
            motion_stats["bg_motion"].append(0.0)
            continue

        # 4. 分析拟合结果
        # mask 是一个 (N, 1) 的数组，值为 1 表示该点符合矩阵变换规律（内点/背景）
        inliers_count = np.sum(mask)
        total_pts = len(mask)
        inlier_ratio = inliers_count / total_pts

        # 计算这些“内点”（即背景）的实际平均运动幅度
        # 这就是真实的相机移动量，排除了前景独立运动物体的干扰
        if inliers_count > 0:
            valid_mask = mask.ravel() == 1
            valid_u = u_sub.ravel()[valid_mask]
            valid_v = v_sub.ravel()[valid_mask]
            bg_motion_magnitude = np.mean(np.sqrt(valid_u**2 + valid_v**2))
        else:
            bg_motion_magnitude = 0.0

        # 5. 综合判定：
        # 必须有足够比例的点（比如50%）遵循同一套运动规律，并且这个运动幅度大于噪点阈值
        is_moving = (inlier_ratio >= inlier_thresh) and (
            bg_motion_magnitude >= motion_thresh
        )

        camera_moving.append(is_moving)
        motion_stats["inlier_ratio"].append(round(float(inlier_ratio), 4))
        motion_stats["bg_motion"].append(round(float(bg_motion_magnitude), 4))

    return camera_moving, motion_stats


@torch.no_grad()
def inference_flow_sparkle(
    model,
    inference_csv,
    video_root,
    num_shards=None,
    shard_index=None,
    output_path="output",
    padding_factor=8,
    inference_size=None,
    save_flo_flow=False,  # save raw flow prediction as .flo
    attn_type="swin",
    attn_splits_list=None,
    corr_radius_list=None,
    prop_radius_list=None,
    num_reg_refine=1,
    pred_bidir_flow=False,
    pred_bwd_flow=False,
    fwd_bwd_consistency_check=False,
    save_video=False,
    concat_flow_img=False,
):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    fixed_inference_size = inference_size
    transpose_img = False

    dset = BaseVideoDataset(
        csv_path=inference_csv,
        video_root=video_root,
        num_shards=num_shards,
        shard_index=shard_index,
    )

    output_path = os.path.join(output_path, f"shards_{num_shards}_{shard_index}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ok, fail = 0, 0
    pbar = tqdm(range(len(dset)))
    with open(output_path, "a") as f:
        for idx in pbar:
            try:
                video, video_path_abs, video_path_rel = dset[idx]
            except Exception as e:
                fail += 1
                pbar.set_postfix(ok=ok, fail=fail)
                print(f"Read error, skipping: {e}")
                continue

            num_frames = video.shape[0]

            video_info = {
                "video_path": video_path_rel,
                "mean_motion": [],
                "mean_relative_motion": [],
            }

            video_flow = []

            for test_id in range(0, num_frames - 1):
                # 直接从 numpy 数组中通过索引切片拿出相邻的两帧
                image1 = video[test_id]  # shape: (320, 576, 3)
                image2 = video[test_id + 1]  # shape: (320, 576, 3)

                image1 = image1.astype(np.uint8)
                image2 = image2.astype(np.uint8)

                if len(image1.shape) == 2:  # gray image
                    image1 = np.tile(image1[..., None], (1, 1, 3))
                    image2 = np.tile(image2[..., None], (1, 1, 3))
                else:
                    image1 = image1[..., :3]
                    image2 = image2[..., :3]

                image1 = (
                    torch.from_numpy(image1)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                image2 = (
                    torch.from_numpy(image2)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

                # the model is trained with size: width > height
                if image1.size(-2) > image1.size(-1):
                    image1 = torch.transpose(image1, -2, -1)
                    image2 = torch.transpose(image2, -2, -1)
                    transpose_img = True

                nearest_size = [
                    int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor,
                ]

                # resize to nearest size or specified size
                inference_size = (
                    nearest_size
                    if fixed_inference_size is None
                    else fixed_inference_size
                )

                assert isinstance(inference_size, list) or isinstance(
                    inference_size, tuple
                )
                ori_size = image1.shape[-2:]

                # resize before inference
                if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                    image1 = F.interpolate(
                        image1, size=inference_size, mode="bilinear", align_corners=True
                    )
                    image2 = F.interpolate(
                        image2, size=inference_size, mode="bilinear", align_corners=True
                    )

                if pred_bwd_flow:
                    image1, image2 = image2, image1

                results_dict = model(
                    image1,
                    image2,
                    attn_type=attn_type,
                    attn_splits_list=attn_splits_list,
                    corr_radius_list=corr_radius_list,
                    prop_radius_list=prop_radius_list,
                    num_reg_refine=num_reg_refine,
                    task="flow",
                    pred_bidir_flow=pred_bidir_flow,
                )

                flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

                # resize back
                if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                    flow_pr = F.interpolate(
                        flow_pr, size=ori_size, mode="bilinear", align_corners=True
                    )
                    flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                    flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                if transpose_img:
                    flow_pr = torch.transpose(flow_pr, -2, -1)

                # flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

                flow = flow_pr[0]  # [2 x H x W]
                motion_info = compute_dual_motion_features(flow)

                video_info["mean_motion"].append(motion_info["mean_motion"])
                video_info["mean_relative_motion"].append(
                    motion_info["mean_relative_motion"]
                )

                video_flow.append(flow)

            video_flow = torch.stack(video_flow).cpu().numpy()
            camera_moving, motion_stats = detect_camera_motion_from_flow(
                video_flow, step=1
            )
            video_info["camera_moving"] = [int(x) for x in camera_moving]
            video_info["motion_stats"] = motion_stats

            f.write(json.dumps(video_info) + "\n")
            f.flush()
            ok += 1
            pbar.set_postfix(ok=ok, fail=fail)

    print(f"Done. ok={ok}, fail={fail}")
