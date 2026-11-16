"""W145 script-only sensitivity and source-conditioned measurement qualification.

This is NOT a pose corrector. Unqualified measurements are null, never held or
smoothed into an Action. Geometric sensitivity is not calibrated pose trust.
"""
from __future__ import annotations
import numpy as np
from motion.torso_frame import torso_frames
from app.experimental_scene_anchored_trace import rotation_angle

AXIS_JOINTS=(0,1,4,8,11,14)


def dimensions(j):
    j=np.asarray(j,float);t=j[8]-j[0];length=np.linalg.norm(t)
    if not np.isfinite(j[list(AXIS_JOINTS)]).all() or length<=1e-12:
        return {'trunk_m':float(length),'valid':False}
    u=t/length;result={'trunk_m':float(length),'valid':True}
    for name,l,r in [('pelvis',1,4),('thorax',11,14)]:
        v=j[r]-j[l];n=np.linalg.norm(v);p=v-u*np.dot(u,v);pn=np.linalg.norm(p)
        result[name]={'side_m':float(n),'projected_side_m':float(pn),
                      'sin_axis_angle':float(pn/max(n,1e-30)),
                      'angle_to_trunk_deg':float(np.degrees(np.arctan2(pn,np.dot(u,v)))),
                      'inverse_projected_length_per_m':float(1/max(pn,1e-30))}
    return result


def differential_frame(j,dj,side):
    """Analytic differential of the frozen Gram-Schmidt frame, not a new formula.

    Returns angular tangent (local axes), radians per unit perturbation.
    """
    j=np.asarray(j,float);dj=np.asarray(dj,float)
    t=j[8]-j[0];dt=dj[8]-dj[0];L=np.linalg.norm(t)
    if L<=1e-12:return None
    u=t/L;du=(np.eye(3)-np.outer(u,u))@dt/L
    l,r=(1,4) if side=='pelvis' else (11,14)
    v=j[r]-j[l];dv=dj[r]-dj[l];P=np.eye(3)-np.outer(u,u)
    p=P@v;n=np.linalg.norm(p)
    if n<=1e-12:return None
    x=p/n;dp=P@dv-(np.outer(du,u)+np.outer(u,du))@v
    dx=(np.eye(3)-np.outer(x,x))@dp/n
    y=np.cross(u,x);dy=np.cross(du,x)+np.cross(u,dx)
    R=np.stack((x,y,u),1);dR=np.stack((dx,dy,du),1)
    omega=R.T@dR;omega=(omega-omega.T)/2
    return np.array([omega[2,1],omega[0,2],omega[1,0]])


def jacobian(j,side):
    columns=[]
    for k in AXIS_JOINTS:
        for axis in range(3):
            dj=np.zeros((17,3));dj[k,axis]=1
            omega=differential_frame(j,dj,side)
            if omega is None:return None
            columns.append(omega)
    return np.stack(columns,1)


def independent_2d_midpoints(x,valid):
    """Update canonical derived points after a declared direct-joint intervention.

    Virtual pelvis = hips midpoint; thorax/neck = shoulders midpoint; spine =
    midpoint of pelvis and neck. Confidence is minimum and validity conjunction.
    """
    x=np.asarray(x,np.float32).copy();valid=np.asarray(valid,bool).copy()
    for out,a,b in [(0,1,4),(8,11,14),(9,11,14),(7,0,8)]:
        valid[out]=bool(valid[a] and valid[b])
        if valid[out]:
            x[out,:2]=(x[a,:2]+x[b,:2])/2
            x[out,2]=min(x[a,2],x[b,2])
        else:x[out]=0
    return x,valid


def qualify(joints,xy_px,valid,texture_residual,texture_valid,
            sigma_m=.002,max_condition_deg=5.,texture_max_px=3.,
            unsupported_turn_deg=10.,quiet_source_px=2.):
    """Qualify measurements; NO latent pose estimate is propagated across gaps.

    A rapid turn is allowed when source motion supports it. A large rotation
    with quiet source is ambiguous (could be a real hidden turn), not labelled
    false with certainty. Source texture is evidence about points, not anatomy.
    """
    joints=np.asarray(joints,float);xy_px=np.asarray(xy_px,float);valid=np.asarray(valid,bool)
    out=[];previous=None
    for i,j in enumerate(joints):
        Rt,Rp=torso_frames(j);record={'frame':i,'pelvis':{},'thorax':{}}
        for name,R,idx in [('pelvis',Rp,[1,4]),('thorax',Rt,[11,14])]:
            reasons=[];J=jacobian(j,name)
            uncertainty=float(np.degrees(np.linalg.svd(J,compute_uv=False)[0]*sigma_m)) if J is not None else None
            if R is None:reasons.append('FRAME_DEGENERATE')
            if uncertainty is None or uncertainty>max_condition_deg:reasons.append('WEAK_GEOMETRY')
            needed=idx+[0,8]
            if not valid[i,needed].all():reasons.append('MISSING_OBSERVATION')
            if i:
                pairvalid=valid[i,idx]&valid[i-1,idx]
                if not pairvalid.all():reasons.append('TEMPORAL_IDENTITY_UNOBSERVED')
                tv=np.asarray(texture_valid[i-1],bool)[idx]
                if not tv.all():reasons.append('TEXTURE_UNAVAILABLE')
                residual=np.asarray(texture_residual[i-1],float)[idx]
                if np.any(tv & (residual>texture_max_px)):reasons.append('POINT_TEXTURE_DISAGREEMENT')
                prior=previous[0 if name=='thorax' else 1]
                if prior is not None and R is not None and pairvalid.all():
                    innovation=float(rotation_angle(prior,R))
                    source_motion=float(np.linalg.norm(xy_px[i,idx]-xy_px[i-1,idx],axis=1).max())
                    if innovation>unsupported_turn_deg and source_motion<quiet_source_px:
                        reasons.append('DEPTH_TURN_UNCONFIRMED')
            record[name]={'status':'UNQUALIFIED' if reasons else 'QUALIFIED_MEASUREMENT',
                          'reasons':reasons,'geometric_2mm_sensitivity_deg':uncertainty,
                          'R':R.tolist() if R is not None and not reasons else None}
        # Always compare consecutive raw measurements; never silently create a
        # track from the last accepted value or force an orientation branch.
        previous=(Rt,Rp);out.append(record)
    return out
