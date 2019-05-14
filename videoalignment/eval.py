# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import itertools

import numpy as np
import progressbar
import torch
from torch.utils.data import DataLoader

from . import datasets, models
from .datasets import batch, is_the_same_pair
from .utils import get_device


def segment_pr(model, dataset, args, phase, crop_before_score=False, merge_mode=None):
    device = get_device(model)
    print("Computing segment Precision and Recall")
    model.eval()

    # Precompute query descriptors
    query_dataset = dataset(phase, args, get_single_videos=True, pad=False)
    query_dataloader = DataLoader(
        query_dataset, batch_size=1, num_workers=min(args.b_s, 12)
    )
    query_iter_dl = iter(query_dataloader)

    query_fvs = []
    query_lengths = []
    bar = progressbar.ProgressBar()
    print("Query Fvs extraction")
    for it, (ts, xs) in enumerate(bar(query_iter_dl)):
        with torch.no_grad():
            ts = ts.float().to(device)
            xs = xs.float().to(device)
            query_fvs.append(model.single_fv(ts, xs).detach().cpu().numpy())
        query_lengths.append(ts.shape[1])
    query_fvs = np.concatenate(query_fvs, 0)

    # Precompute video descriptors
    videos_dataset = dataset(phase, args, get_entire_videos=True, pad=False)
    entire_videos = set(v["video"] for v in query_dataset.videos)
    entire_videos = [
        next(v for v in videos_dataset.videos if v["video"] == ev)
        for ev in entire_videos
    ]
    videos_dataset.videos = entire_videos
    videos_dataloader = DataLoader(
        videos_dataset, batch_size=1, num_workers=min(args.b_s, 12)
    )
    videos_iter_dl = iter(videos_dataloader)

    video_fvs = []
    video_lengths = []
    bar = progressbar.ProgressBar()
    print("Videos Fvs extraction")
    for it, (ts, xs) in enumerate(bar(videos_iter_dl)):
        with torch.no_grad():
            ts = ts.float().to(device)
            xs = xs.float().to(device)
            video_fvs.append(model.single_fv(ts, xs).detach().cpu().numpy())
        video_lengths.append(xs.shape[1])
    video_fvs = np.concatenate(video_fvs, 0)

    # Loop over queries
    probas = []
    labels = []
    for qv, query_fv, query_length in zip(
        query_dataset.videos, query_fvs, query_lengths
    ):
        with torch.no_grad():
            print("Processing", qv)
            query_fv = torch.from_numpy(query_fv).unsqueeze(0).float().to(device)

            for v, video_fv, video_length in zip(
                entire_videos, video_fvs, video_lengths
            ):
                if v["video"] == qv["video"]:
                    continue
                video_fv = torch.from_numpy(video_fv).unsqueeze(0).float().to(device)
                all_offsets = torch.arange(-video_length, 0).unsqueeze(0).to(device)
                delta = model.score_pair(query_fv, video_fv, all_offsets)
                score, delta = torch.max(delta, 1)
                score = score.detach().cpu().numpy()[0]

                delta = delta - video_length
                delta = -delta.data.cpu().numpy()[0]
                delta = delta / videos_dataset.fps
                probas.append(score)
                query_segment = np.around(np.arange(qv["begin"], qv["end"], 0.01), 2)
                video_segment = np.around(
                    np.arange(delta, delta + qv["end"] - qv["begin"], 0.01), 2
                )

                ops = [
                    op
                    for op in query_dataset.overlapping_pairs
                    if (
                        op["videos"][0]["video"] == qv["video"]
                        and op["videos"][1]["video"] == v["video"]
                    )
                    or (
                        op["videos"][1]["video"] == qv["video"]
                        and op["videos"][0]["video"] == v["video"]
                    )
                ]
                found = False
                could_be_fn = False
                for opi in ops:
                    if opi["videos"][0]["video"] == qv["video"]:
                        os_q = opi["videos"][0]
                        os_v = opi["videos"][1]
                    else:
                        os_q = opi["videos"][1]
                        os_v = opi["videos"][0]
                    this_query_segment = np.around(
                        np.arange(os_q["begin"], os_q["end"], 0.01), 2
                    )
                    this_video_segment = np.around(
                        np.arange(os_v["begin"], os_v["end"], 0.01), 2
                    )

                    inter_size_q = np.intersect1d(
                        query_segment, this_query_segment
                    ).size
                    inter_size_v = np.intersect1d(
                        video_segment, this_video_segment
                    ).size

                    if inter_size_q:
                        could_be_fn = True

                    if inter_size_q > 0 and inter_size_v > 0:
                        label = 1
                        found = True
                        break

                if not found:
                    label = 0
                    if could_be_fn:
                        probas.append(0)
                        labels.append(1)

                labels.append(label)

    return probas, labels


def map(model, dataset, args, phase, **kwargs):
    if dataset == datasets.EVVE:
        return map_evve(model, dataset, args, **kwargs)
    else:
        return map_localization(model, dataset, args, phase)


def map_evve(
    model, dataset, args, query_expansion="", temporal_consistency=False, N1=10, N2=2000
):
    device = get_device(model)

    def score_ap_from_ranks_1(ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0

        return ap

    print("Computing mAP...")
    model.eval()

    # Get feature vectors for every video
    dataset_obj = dataset("all", args, get_single_videos=True, pad=False)

    dataloader = DataLoader(dataset_obj, batch_size=1, num_workers=16)
    iter_dl = iter(dataloader)

    fvs = []
    fvs_length = []
    bar = progressbar.ProgressBar()
    print("Extracting feature vectors...")
    for it, (ts, xs) in enumerate(bar(iter_dl)):
        with torch.no_grad():
            ts = ts.float().to(device)
            xs = xs.float().to(device)
            fvs.append(model.single_fv(ts, xs).data.cpu().numpy())
        fvs_length.append(ts.shape[1])
    fvs = np.concatenate(fvs, 0)
    fvs_length = np.asarray(fvs_length)

    # Compute pairwise scores inside each event
    length = dataset_obj.length
    all_offsets = torch.arange(-length, length).unsqueeze(0).to(device)
    events = list(set(v["event"] for v in dataset_obj.videos))
    results = []
    db_ids = [
        i
        for i in range(len(dataset_obj.videos))
        if dataset_obj.videos[i]["split"] == "database"
    ]
    dbs = [dataset_obj.videos[i] for i in db_ids]
    fvs_db = fvs[db_ids]
    fvs_db_len = fvs_length[db_ids]
    for e in events:
        this_results = []
        this_ids = [
            i
            for i in range(len(dataset_obj.videos))
            if dataset_obj.videos[i]["event"] == e
        ]
        this_query_ids = [
            i for i in this_ids if dataset_obj.videos[i]["split"] == "query"
        ]
        this_queries = [dataset_obj.videos[i] for i in this_query_ids]
        this_fvs_query = fvs[this_query_ids]
        this_fvs_query_len = fvs_length[this_query_ids]

        for query, query_fv, query_len in zip(
            this_queries, this_fvs_query, this_fvs_query_len
        ):
            scores = []
            offsets = []
            for db_fv, db_len in zip(
                batch(fvs_db, args.b_s // 4), batch(fvs_db_len, args.b_s // 4)
            ):
                with torch.no_grad():
                    fv_a = torch.from_numpy(query_fv).unsqueeze(0).float().to(device)
                    fv_b = torch.from_numpy(db_fv).float().to(device)

                    if args.model == models.CTE:
                        max_len = max(query_len, max(db_len))
                        score, offset = torch.max(
                            model.score_pair(fv_a, fv_b, all_offsets, max_len), -1
                        )
                    else:
                        score, offset = torch.max(
                            model.score_pair(fv_a, fv_b, all_offsets), -1
                        )
                    offset = offset.detach().cpu().numpy()
                    score = score.detach().cpu().numpy()
                scores.append(score)
                offsets.append(offset - length)

            scores = np.concatenate(scores, 0)
            offsets = np.concatenate(offsets, 0)
            scores = np.argsort(scores)[::-1]

            if query_expansion in ["aqe", "don"]:
                scores_exp = scores[:N1]
                offsets_exp = offsets[scores_exp]
                fvs_db_exp = [fvs_db[i] for i in scores_exp]

                # Find whether the returned items are temporally consistent
                if temporal_consistency:
                    temporal_scores = [0 for _ in range(N1)]
                    for i_a, i_b in itertools.combinations(range(N1), 2):
                        with torch.no_grad():
                            fv_a = (
                                torch.from_numpy(fvs_db_exp[i_a])
                                .unsqueeze(0)
                                .float()
                                .to(device)
                            )
                            fv_b = (
                                torch.from_numpy(fvs_db_exp[i_b])
                                .unsqueeze(0)
                                .float()
                                .to(device)
                            )
                            _, offset = torch.max(
                                model.score_pair(fv_a, fv_b, all_offsets), -1
                            )
                            offset = offset.detach().cpu().numpy()[0] - length
                        this_score = np.abs(offsets[i_a] - offsets[i_b] + offset)
                        temporal_scores[i_a] += this_score
                        temporal_scores[i_b] += this_score

                    temporal_consistent_ids = np.argsort(temporal_scores)[: N1 // 2]
                    offsets_exp = offsets_exp[temporal_consistent_ids]
                    fvs_db_exp = [fvs_db_exp[i] for i in temporal_consistent_ids]

                for i, (fv_db_exp, offset_exp) in enumerate(
                    zip(fvs_db_exp, offsets_exp)
                ):
                    fv_db_exp = (
                        torch.from_numpy(fv_db_exp).unsqueeze(0).float().to(device)
                    )
                    fv_db_exp = model.shift_fv(fv_db_exp, offset_exp)
                    fvs_db_exp[i] = fv_db_exp.data.cpu().numpy()[0]

                query_exp_fv = (query_fv + sum(fvs_db_exp)) / (N1 + 1)

                if query_expansion == "don":
                    scores_exp = scores[:N2]
                    fvs_db_exp = [fvs_db[i] for i in scores_exp]
                    query_exp_fv = query_exp_fv - sum(fvs_db_exp) / N2

                scores = []
                for db_fv in batch(fvs_db, args.b_s // 4):
                    with torch.no_grad():
                        fv_a = (
                            torch.from_numpy(query_exp_fv)
                            .unsqueeze(0)
                            .float()
                            .to(device)
                        )
                        fv_b = torch.from_numpy(db_fv).float().to(device)
                        score = (
                            torch.max(model.score_pair(fv_a, fv_b, all_offsets), -1)[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    scores.append(score)
                scores = np.concatenate(scores, 0)
                scores = np.argsort(scores)[::-1]

            sorted_dbs = [dbs[i] for i in scores]
            pos_ranks = []  # ranks of TPs (0-based)
            rank_shift = 0  # adjust rank to ignore nulls
            pos = [
                v for v in dbs if v["pos_null"] == 1 and v["event"] == query["event"]
            ]
            null = [
                v for v in dbs if v["pos_null"] == -1 and v["event"] == query["event"]
            ]
            for rank, db in enumerate(sorted_dbs):
                if db in pos:
                    pos_ranks.append(rank - rank_shift)
                elif db in null:
                    rank_shift += 1

            ap = score_ap_from_ranks_1(pos_ranks, len(pos))
            this_results.append(ap)
        print("%s\t%f" % (e, sum(this_results) / len(this_results)))
        results.extend(this_results)

    final_map = sum(results) / len(results)
    print("Overall mAP:", final_map)
    return final_map


def map_localization(model, dataset, args, phase):
    device = get_device(model)
    print("Computing mAP...")
    model.eval()

    # Get feature vectors for every video
    dataset_obj = dataset(phase, args, get_single_videos=True)
    ts, xs = dataset_obj[0]
    dataloader = DataLoader(
        dataset_obj, batch_size=args.b_s // 4, num_workers=min(3 * args.b_s // 4, 12)
    )
    iter_dl = iter(dataloader)

    fvs = []
    with torch.no_grad():
        for it, (ts, xs) in enumerate(iter_dl):
            ts = ts.float().to(device)
            xs = xs.float().to(device)
            fvs.append(model.single_fv(ts, xs).data.cpu().numpy())
        fvs = np.concatenate(fvs, 0)

    # Compute pairwise scores
    all_pairs = dataset_obj.all_pairs
    iter_comb = list(itertools.combinations(fvs, 2))
    scores = []

    length = dataset_obj.length
    all_offsets = torch.arange(-length, length).unsqueeze(0).to(device)

    with torch.no_grad():
        for it, fvs in enumerate(batch(iter_comb, args.b_s // 4)):
            fv_a = np.asarray([fv[0] for fv in fvs])
            fv_b = np.asarray([fv[1] for fv in fvs])
            fv_a = torch.from_numpy(fv_a).float().to(device)
            fv_b = torch.from_numpy(fv_b).float().to(device)
            scores.append(
                torch.max(model.score_pair(fv_a, fv_b, all_offsets), -1)[0]
                .detach()
                .cpu()
                .numpy()
            )
    scores = np.concatenate(scores, 0)

    # for each query....
    all_pairs_dict = dict()
    for i, p in enumerate(all_pairs):
        for k in (frozenset(p["videos"][0].items()), frozenset(p["videos"][1].items())):
            if k not in all_pairs_dict.keys():
                all_pairs_dict[k] = [i]
            else:
                all_pairs_dict[k].append(i)

    map = 0
    for v_i, v in enumerate(dataset_obj.videos):
        ap = 0
        tp = 0
        rs_indexes = all_pairs_dict[frozenset(v.items())]
        rs = [all_pairs[i] for i in rs_indexes]
        rs_scores = np.asarray([scores[i] for i in rs_indexes])
        rs_index = np.argsort(rs_scores)[::-1][:50]

        for seen, idx in enumerate(rs_index):
            if any(
                op
                for op in dataset_obj.overlapping_pairs
                if is_the_same_pair(op, rs[idx])
            ):
                if seen == 0:
                    precision_0 = 1
                else:
                    precision_0 = tp / seen
                precision_1 = (tp + 1) / (seen + 1)
                ap += (precision_0 + precision_1) / 2
                tp += 1
        if tp > 0:
            ap = ap / tp
        map += ap
    return map / len(dataset_obj.videos)


def localization_errors(model, dataset, args, phase):
    device = get_device(model)
    print("Computing localization error...")
    model.eval()

    # Get feature vectors for every video
    dataset_obj = dataset(phase, args, get_single_videos=True, pad=False)
    dataloader = DataLoader(
        dataset_obj, batch_size=1, num_workers=min(3 * args.b_s // 4, 12)
    )
    iter_dl = iter(dataloader)

    fvs = []

    for it, (ts, xs) in enumerate(iter_dl):
        with torch.no_grad():
            ts = ts.float().to(device)
            xs = xs.float().to(device)
            fvs.append(model.single_fv(ts, xs).data.cpu().numpy())
    fvs = np.concatenate(fvs, 0)

    # Compute pairwise scores
    all_pairs = dataset_obj.all_pairs
    iter_comb = list(itertools.combinations(fvs, 2))
    scores = []
    length = dataset_obj.length
    all_offsets = torch.arange(-length, length).unsqueeze(0).to(device)

    for it, fvs in enumerate(batch(iter_comb, 1)):
        with torch.no_grad():
            fv_a = np.asarray([fv[0] for fv in fvs])
            fv_b = np.asarray([fv[1] for fv in fvs])

            fv_a = torch.from_numpy(fv_a).float().to(device)
            fv_b = torch.from_numpy(fv_b).float().to(device)
            scores.append(model.score_pair(fv_a, fv_b, all_offsets).data.cpu().numpy())
    scores = np.concatenate(scores, 0)

    # for each query....
    errors = []
    all_pairs_dict = dict()
    for i, p in enumerate(all_pairs):
        for k in (frozenset(p["videos"][0].items()), frozenset(p["videos"][1].items())):
            if k not in all_pairs_dict.keys():
                all_pairs_dict[k] = [i]
            else:
                all_pairs_dict[k].append(i)

    for v_i, v in enumerate(dataset_obj.videos):
        rs_indexes = all_pairs_dict[frozenset(v.items())]
        rs = [all_pairs[i] for i in rs_indexes]
        rs_scores = np.asarray([scores[i] for i in rs_indexes])
        rs_index = np.argsort(np.max(rs_scores, -1))[::-1]
        rs_index = rs_index[0]
        rs = rs[rs_index]
        det_offset = (
            np.argmax(rs_scores[rs_index]) - dataset_obj.length
        ) / dataset_obj.fps

        try:
            op = next(
                op for op in dataset_obj.overlapping_pairs if is_the_same_pair(op, rs)
            )
            if op["videos"] == rs["videos"]:
                gt_offset = -op["offset"]
            else:
                gt_offset = op["offset"]
            errors.append(abs(det_offset - gt_offset))
        except:
            errors.append(float("inf"))

    errors = np.array(errors)

    def better_than_t(t):
        return np.sum(errors < t) / errors.size

    for t in [0.1, 1, 10]:
        print(f"Localization error <{t}s: {better_than_t(t)*100:.2f}%")
    return errors
