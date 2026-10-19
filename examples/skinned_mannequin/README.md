# BaseRig skinned mannequin

`BaseRig_Skinned.fbx` is the original `../BaseRig.fbx` armature with an simple flat-shaded low-poly mannequin skin. Use this file as the rig input when a visible character is needed. The original bare rig is unchanged. `BaseRig_Skinned.blend` contains editable skin weights, the armature, and a CPU render studio. The FBX contains only the armature and skin, without cameras, lights, or animation.

A small nose and forward-projecting feet distinguish front (-Y in Blender) from back (+Y). Anatomical muscle details are omitted; the hands are simplified mitten shapes. The skin has 698 vertices and 842 polygons. The model follows the existing rest pose. All 51 original bones, their names and parents are preserved; no head/pelvis/control bones were added. The head follows `neck_02`; the pelvis blends the lower spine and thighs. All vertices have one to four normalized influences. The slate-blue material requires no textures.

## Rebuild and validate

From the repository root, using Blender 5.1.2:

```sh
blender -b --python scripts/build_skinned_mannequin.py
blender -b --python scripts/verify_skinned_mannequin.py
```

On macOS, the Blender executable can be `/Applications/Blender.app/Contents/MacOS/Blender`. Both scripts are platform-independent Blender Python scripts. Generation overwrites this directory's generated files. Rendering explicitly uses CPU.

`validation.json` records source/output hashes, imported bone-rest matrix agreement, normalized weights and five individual 45-degree local-X deformation checks. `preview_front.png`, `preview_back.png`, and `preview_three_quarter.png` show the rest pose. `preview_pose_check.png` is an additional visual spot check (left forearm -55°, right calf +55°, spine_03 +10°, local X), not an animation included in the asset.

This is a procedural mannequin for animation inspection, with overlapping low-poly body sections joined into one mesh object. It is not production character retopology, and extreme poses may need weight painting or corrective shapes. The hand bones are preserved, but individual fingers are not modeled. Joint checks establish that skin follows bones, not collision-free behavior across every pose. No perception benchmark, training, GPU experiment, or full AnimCV pipeline evaluation was performed for this asset.
