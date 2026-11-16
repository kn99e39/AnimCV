"""W144 synchronized source + actual Blender renders. Diagnostic views, no camera fit."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import w144_harness as H
from w144_analyze import JOINTS
from app import experimental_scene_anchored_trace as T


def fit_panel(im,width,height):
    """Letterbox review crops; never stretch source anatomy."""
    import cv2
    h,w=im.shape[:2];scale=min(width/w,height/h)
    resized=cv2.resize(im,(max(1,round(w*scale)),max(1,round(h*scale))))
    out=np.full((height,width,3),24,np.uint8)
    y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return out


def main():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H.verify();r=H.load(H.OUT/'analysis.json');edges=H.load(H.OUT/'edge_trajectories.json')
    c=H.load(H.CAM);good=[f['orientation']['status']=='ORIENTATION_ESTIMATED' for f in c['frames']]
    validpairs=np.array(good[:-1])&good[1:]
    camera=[];composed=[]
    poses=np.load(H.OBS)['predictions'][79:319]
    from motion.torso_frame import torso_frames
    S=np.array([[1,0,0],[0,0,1],[0,-1,0.]]) # optical -> canonical
    Rc=[S@np.array(f['orientation']['R_WC'])@S.T if g else None for f,g in zip(c['frames'],good)]
    Rp=[torso_frames(p)[1] for p in poses]
    for i in range(239):
        if validpairs[i]:
            camera.append(float(T.rotation_angle(Rc[i],Rc[i+1])))
            composed.append(float(T.rotation_angle(Rc[i]@Rp[i],Rc[i+1]@Rp[i+1])))
    camera_rep={'status':'nominal orientation-only hypothesis, not measured metric camera pose',
        'coverage':int(sum(good)),'valid_adjacent_pairs':int(validpairs.sum()),
        'rotation_increment_deg':T.summary(camera),'root_rotation_with_nominal_camera_deg':T.summary(composed),
        'background_flow_all_edges_px':T.summary([e['flow_raw_median_px'] for e in c['edges'] if e.get('flow_raw_median_px') is not None]),
        'background_keypoints':T.summary([f['evidence']['n_kp'] for f in c['frames']]),
        'translation':'unresolved, never substituted by zero; no S1 reprojection gate',
        'baseline_camera':'absent in authentic source-to-Action path; constant rotation reference is identity convention'}
    H.write(H.OUT/'camera_reference.json',camera_rep)
    x=np.arange(2,241);fig,ax=plt.subplots(5,1,figsize=(14,12),sharex=True)
    for key in ('pelvis','thorax','left_shoulder','right_shoulder'):
        ax[0].plot(x,[e[f'obs_{key}_px'] for e in edges],label=key,lw=1)
    ax[0].set_ylabel('2D step (px)');ax[0].legend(ncol=4);ax[0].set_title('S1: observations and hypotheses kept separate; no calibrated reprojection')
    for key in ('pelvis_proxy','torso','shoulder_l','shoulder_r'):
        ax[1].plot(x,[1000*e[f'world_{key}_m'] for e in edges],label=key,lw=1)
    ax[1].set_ylabel('Blender step (mm)');ax[1].legend(ncol=4)
    for key in ('object','spine_01','inferred_thorax_Rt'):
        ax[2].plot(x,[e[f'rotation_{key}_deg'] for e in edges],label=key,lw=1)
    ax[2].set_ylabel('rotation step (deg)');ax[2].legend(ncol=3)
    for key in ('object_rotation','bone:spine_01','object_rotation+spine_01'):
        ax[3].plot(x,[1000*e[f'hold_{key}_shoulder_l_m'] for e in edges],label='hold '+key,lw=1)
    ax[3].set_ylabel('diagnostic step (mm)');ax[3].legend(ncol=3)
    sc=H.load(H.OUT/'source_controls.json')
    for key in ('left_shoulder','right_shoulder'):
        k=JOINTS.index(key)
        ax[4].plot(x,[e['landmark_texture_residual_px'][k] for e in sc['edges']],label=key+' vs LK texture',lw=1)
    ax[4].set_ylabel('texture residual (px)');ax[4].legend(ncol=2);ax[4].set_xlabel('destination Action frame (source = Action + 78)')
    for a in ax:
        for lo,hi in r['selection']['fixed']:a.axvspan(lo,hi,color='grey',alpha=.06)
        a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(H.OUT/'trajectory_attribution.png',dpi=140);plt.close(fig)
    # Composite the original source frame with actual fresh Blender mesh renders.
    obs=np.load(H.OBS)['input_2d'];valid=np.load(H.OBS)['input_valid']
    review=ROOT/'user_eval/W144_scene_anchored';review.mkdir(parents=True,exist_ok=False)
    movie=review/'S1_source_baseline_30fps.mp4'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),30,(1408,720))
    if not writer.isOpened():raise RuntimeError('video writer unavailable')
    cap=cv2.VideoCapture(str(H.CLIP));sheets={}
    windows={**{f'peak_{i}':[max(1,i-3),min(240,i+3)] for i in r['selection']['top_shoulder_destinations']},
        'quiet':r['selection']['quiet'],'large_source':r['selection']['large_source_wrist'],'uncertain':r['selection']['uncertain'],
        'entry':[168,175],'historical244':[174,180],'exit':[196,203],'start':[1,12],'end':[229,240]}
    wanted=set()
    for lo,hi in windows.values():wanted.update((lo,(lo+hi)//2,hi))
    wanted.update((21,22,23,25,26,27,176,177,178,171,172,199,200))
    for source in range(319):
        ok,im=cap.read();assert ok
        if source<79:continue
        action=source-78;canvas=np.full((720,1408,3),23,np.uint8)
        canvas[50:530,:640]=im
        for k in (0,1,4,8,10,11,12,13,14,15,16,3,6):
            p=tuple(np.round(obs[source,k,:2]*[640,480]).astype(int)+[0,50])
            if valid[source,k]:cv2.circle(canvas,p,3,(80,230,240) if k<14 else (230,180,60),1)
        for col,view in enumerate(('world_front','world_side')):
            ren=cv2.imread(str(H.OUT/f'renders/{view}_f{action:04d}.png'));assert ren is not None
            canvas[80:464,640+col*384:1024+col*384]=ren
        def text(s,x,y,scale=.56,color=(230,230,230)):
            cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        text(f'S1 original | source {source} | Action {action} | {source/30:.3f}s',10,28)
        text('A: original Blender Base | world front',650,28,.5)
        text('A: same Action | world side',1032,28,.5)
        text('30 fps, full 240 frames | no reconstruction candidate justified',12,560,.68)
        if action>1:
            e=edges[action-2]
            text(f"Object travel: 0 mm | Object rotation: {e['rotation_object_deg']:.2f} deg | spine_01: {e['rotation_spine_01_deg']:.2f} deg",12,592)
            text(f"World L shoulder: {e['world_shoulder_l_m']*1000:.1f} mm | pelvis proxy: {e['world_pelvis_proxy_m']*1000:.1f} mm",12,620)
            text(f"Observed shoulders: L {e['obs_left_shoulder_px']:.2f} px, R {e['obs_right_shoulder_px']:.2f} px | detector tracks, not GT",12,648)
        text('Views are diagnostic orthographic cameras; no source reprojection/calibration claim. Right arm remains at rest.',12,675,.49)
        text('Source: Taco fleur / Wikimedia Commons / CC BY-SA 4.0 | local review derivative; no human acceptance',12,702,.46)
        writer.write(canvas)
        if action in wanted:sheets[action]=canvas.copy()
    cap.release();writer.release()
    # Preserve source IDs, zoom subject without warping the original panel.
    for tag,(lo,hi) in windows.items():
        ids=sorted(set([lo,(lo+hi)//2,hi]+[f for f in (22,23,26,27,171,172,176,177,199,200) if lo<=f<=hi]))
        sheet=np.full((len(ids)*240,960,3),24,np.uint8)
        for row,f in enumerate(ids):
            canvas=sheets[f]
            # Source crop for visibility plus both actual Blender views.
            sheet[row*240:(row+1)*240,:320]=fit_panel(canvas[70:530,180:460],320,240)
            sheet[row*240:(row+1)*240,320:640]=fit_panel(canvas[80:464,640:1024],320,240)
            sheet[row*240:(row+1)*240,640:960]=fit_panel(canvas[80:464,1024:1408],320,240)
            cv2.putText(sheet,f'Action {f} / source {f+78}',(8,row*240+20),0,.55,(255,255,255),1)
        cv2.imwrite(str(review/f'{tag}.jpg'),sheet)
    import shutil
    shutil.copy2(H.OUT/'trajectory_attribution.png',review/'trajectory_attribution.png')
    cap=cv2.VideoCapture(str(movie));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while cap.read()[0]:n+=1
    cap.release()
    H.write(H.OUT/'review_artifacts.json',{'movie':str(movie),'sha256':H.sha(movie),'decoded_frames':n,'fps':fps,
        'seconds':n/fps,'source_frames':[79,318],'action_frames':[1,240],'actual_Blender_renders':480,
        'source_identity':H.sha(H.CLIP),'windows':windows,'agent_native_playback_review':'pending',
        'human_acceptance':False,'camera_reference':camera_rep})
    print('REVIEW',movie,n,fps)


if __name__=='__main__':main()
