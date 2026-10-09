# Bao cao bai nop - Day 23 Sensor Fusion Lab

## Thong tin hoc vien

- Họ tên: Nguyen Minh Duc
- MSSV: 2A202602783
- Email: duc.nm224954@sis.hust.edu.vn
- Link repo (fork): https://github.com/minhduckx2004/K4-L2L3-DAY23-NguyenMinhDuc-2A202602783-SensorFusion
- Commit hash nop (`git rev-parse HEAD`): xem hash final tren LMS sau commit CP6

## Tom tat ket qua

- `fusion_mode` (bat buoc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision `0.9700934579439252`, recall `0.7004048582995951`, tp/fp/fn `519/16/222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse `0.15032268781360134`, matches `502`, sum_sq_err `11.343649056695735`, ghost `0`, miss `239`, mean_confirmed_tracks `2.522613065326633`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse `0.1358667883353908`, matches `502`, sum_sq_err `9.26681165463209`, ghost `0`, miss `239`, mean_confirmed_tracks `2.522613065326633`.
- Giai thich khac biet hai mode, doc RMSE cung so ghep va ghost/miss: ca hai mode co cung `502` matches, `0` ghost va `239` missed_gt_frames, nen so sanh RMSE la hop ly tren cung coverage. Fused giam RMSE tu `0.1503 m` xuong `0.1359 m` (chenh `-0.0145 m`) nho update camera 2D tinh chinh state EKF sau LiDAR. `precision_track = 502/(502+0) = 1.0`; `coverage = 502/519 = 0.9672`. Camera trong lab dung tam hop 2D ground-truth FRONT cong nhieu theo seed, khong phai detector anh that, nen ket qua fused chi chung minh mo hinh fusion trong lab, khong chung minh chat luong camera detector.

Chay tu root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`grade_run.log` co `398` records, dung 2 mode `lidar` va `fused`, frame 0 den 198. Kiem tra moi record deu thoa `matches+ghosts==confirmed` va `matches+misses==valid_gt`.

## Giai thich ngan (Parts E-H - tu viet)

1. Khac biet do lidar 3D va camera 2D trong EKF (`z`, `R`)?
   LiDAR do truc tiep vi tri 3D trong sensor frame, nen `z` co 3 chieu `(x, y, z)` va `R` dung don vi met. Camera do diem anh 2D `(u, v)`, `z` co 2 chieu va `R` dung don vi pixel; ham do camera la pinhole projection phi tuyen nen EKF can Jacobian `H`.
2. Vi sao can gating Mahalanobis truoc khi gan?
   Gating loai cap track-measurement co residual qua lon so voi bat dinh `P` va nhieu do `R`. Mahalanobis tot hon Euclidean vi cung mot khoang cach hinh hoc co the hop ly neu covariance lon, nhung bat hop ly neu covariance nho.
3. Pipeline la track-then-fuse hay fuse-then-track? Chi ra tren log `fusion-run-lab`.
   Pipeline la track-then-fuse: moi frame predict track mot lan, gan/update LiDAR, roi gan/update camera neu co. Log compare tach hai mode `lidar` va `fused`; fused van co cung so confirmed/matches voi LiDAR trong lan chay nay, camera chi tinh chinh state sau khi track da ton tai.
4. Neu camera lech calibration, trieu chung gi tren innovation/residual?
   Projection `h(x)` se bi lech co he thong, innovation pixel se co bias theo mot huong thay vi nhieu zero-mean. Mahalanobis cua camera tang, nhieu cap bi gate loai hoac update camera keo state sai lam RMSE fused xau hon LiDAR.
5. Vi sao `associate_and_update(..., sensor)` can sensor tuong minh o frame rong? Giai thich vi sao lidar quyet dinh score/init/delete con camera chi EKF update.
   Frame rong van can biet pass hien tai la LiDAR hay camera. LiDAR frame rong tao miss trong FOV de tru score va co the xoa track; camera frame rong khong duoc lam thay doi score. Thiet ke lab de LiDAR quyet dinh init/confirm/delete vi detector LiDAR sinh hop 3D, con camera 2D chi bo sung rang buoc do anh cho EKF.
6. Neu dieu kien xac nhan, giu confirmed sau miss, va dieu kien xoa track.
   Track duoc xac nhan khi score `> confirmed_threshold`. Hit LiDAR cong `1/window` toi da 1; miss LiDAR trong FOV tru `1/window`. Track da confirmed van giu state confirmed sau miss don le. Xoa track neu `P[0,0]` hoac `P[1,1] > max_P`, hoac confirmed co score `< delete_threshold`, hoac chua confirmed co score `<= 0`.

## Bonus (khong bat buoc)

Khong.

## Khai bao su dung AI (bat buoc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT Codex.
- Dung cho phan nao (ham, cau hoi, debug): Ho tro cai dat va debug Part E-H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`), chay pytest, chay `fusion-run-lab`, va dien bao cao dua tren artifact.
- Cach ban da kiem tra lai (pytest, chay Waymo, doi chieu cong thuc): Da chay `$env:PYTHONUTF8='1'; python -m pytest student/tests -q` voi `128 passed`; da chay `fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0`; da doi chieu `metrics.json` voi `grade_run.log` va kiem tra identity log.

## Checklist nop

- [x] Part E-H trong `workspace/` da implement; `pytest student/tests -q` khong con `failed`/`xfailed`
- [x] Part A-D: khong sua
- [x] Lan chay cham diem: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Da commit `student/artifacts/metrics*.json` va `student/artifacts/grade_run*.log` (khong sua tay)
- [x] Da dien du file nay, gom khai bao AI
- [x] Khong commit du lieu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` bao san sang nop
- [ ] Da push va nop link repo + commit hash tren LMS
