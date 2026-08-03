# MPI-INF-3DHP 보정 평가 데모

이 디렉터리는 **원본 데이터 없이** MPI-INF-3DHP 기반 정량 평가를 재현하는 방법과 마지막 실행의 요약만 보관한다. 원본 영상·`annot.mat`·camera calibration은 MPI-INF-3DHP의 연구용 라이선스 때문에 이 리포지터리에 넣지 않는다.

## 입력과 실행 범위

- 데이터: 공식 MPI-INF-3DHP `S1/Seq1`, camera 0, 2048×2048, 25fps.
- 회전 구간: 원본 프레임 `3212..3331` (120프레임). GT bilateral axis 변화 폭은 315.33°였다.
- raw 데이터는 사용자가 별도로 받은 로컬 경로에서만 사용한다.

```bash
motion-tool extract-frames --video /path/to/mpi3dhp/S1/Seq1/imageSequence/video_0.avi \
  --out /tmp/animcv-eval/frames --start-frame 3212 --end-frame 3331
motion-tool import-mpi3dhp-ground-truth \
  --annotation /path/to/mpi3dhp/S1/Seq1/annot.mat \
  --calibration /path/to/mpi3dhp/S1/Seq1/camera.calibration --camera-index 0 \
  --start-frame 3212 --end-frame 3331 \
  --pose-out /tmp/animcv-eval/gt_pose.json \
  --lifted-out /tmp/animcv-eval/gt_lifted.json \
  --calibration-out /tmp/animcv-eval/camera.json
```

`--evaluation-ground-truth`는 GT bounding box로 top-down pose 회귀만 분리하는 벤치마크 전용 옵션이다. 제품용 추론/FBX 출력에는 사용하면 안 된다. 이어서 `estimate-pose`, `lift-pose3d`, `estimate-root-motion`, `audit-mpi3dhp-2d`, `audit-mpi3dhp-3d` 명령으로 평가한다.

## 현재 판정

최신 결과는 [latest_summary.json](latest_summary.json)과 [문서 보고서](../../docs/04_MPI_INF_3DHP_보정_평가.md)를 참조한다. 현재는 **게임 제작 파이프라인 채택 불가**다. 2D top-down 회귀는 조건부 통과했지만, 사람 검출 없는 기본 경로와 3D/root-yaw 단계가 모두 실패했다.
