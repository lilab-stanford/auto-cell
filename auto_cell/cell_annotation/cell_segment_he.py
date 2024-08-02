from __future__ import print_function, unicode_literals, absolute_import, division
import pandas as pd
from tqdm import tqdm
from PIL import Image
from csbdeep.utils import Path, normalize

from stardist.models import StarDist2D
from normalization import normalizeStaining
from auto_cell.utils import *


patch_size=75
model = StarDist2D.from_pretrained('2D_versatile_he')
png_dir='./registration_P1'
res_dir='./registration_P1_seg'
os.makedirs(os.path.join(res_dir,'cores'),exist_ok=True)
os.makedirs(os.path.join(res_dir,'texts'),exist_ok=True)
os.makedirs(os.path.join(res_dir,'overlay'),exist_ok=True)
all_pngs=os.listdir(png_dir)
all_pngs.sort()
for img_i in tqdm(all_pngs[:]):
    img_id=img_i.split('.png')[0]
    img=Image.open(os.path.join(png_dir,img_i)).convert('RGB')
    img=np.array(img)
    img = normalizeStaining(img)
    h_img,w_img=img.shape[:2]
    img_nor = normalize(img, 1,99.8, axis=(0,1,2))
    labels, polys = model.predict_instances_big(img_nor, axes='YXC', block_size=2048, min_overlap=128, context=128, n_tiles=(2,2,1),labels_out=False,prob_thresh=0.3)
    roi_info_dict={}
    for inst in range(len(polys['prob'])):
        inst_contour = polys['coord'][inst]
        inst_centroid = polys['points'][inst]
        inst_prob = polys['prob'][inst]
        roi_info_dict[inst] = {'contour': np.flip(np.array(inst_contour).transpose(), axis=1).astype(int), 'prob': inst_prob,'type':0,'centroid':[inst_centroid[1],inst_centroid[0]]}
    overlay=visualize_instances_dict(img, roi_info_dict, draw_dot=True, type_colour={0 : ["nolabel", [0  , 0,   255]]})
    Image.fromarray(overlay, "RGB").save(os.path.join(res_dir,'overlay',img_id+'.png'))
    core_dir=os.path.join(res_dir, 'cores')
    os.makedirs(core_dir, exist_ok=True)
    pngs=[]
    li_img_id = []
    li_cell_id = []
    li_x = []
    li_y = []
    li_prop = []
    li_area = []
    li_len = []
    li_cir = []
    li_sol = []
    for i, [_, inst_info] in enumerate(roi_info_dict.items()):
        i_contour = inst_info["contour"]
        i_centroid=inst_info["centroid"]
        i_prob=inst_info["prob"]
        i_bbox = np.array([[np.min(i_contour[:, 0]), np.min(i_contour[:, 1])]])
        i_bbox = np.concatenate((i_bbox, [[np.max(i_contour[:, 0]), np.max(i_contour[:, 1])]]), axis=0)
        x_cell = i_bbox[0][0]
        y_cell = i_bbox[0][1]
        w_cell = i_bbox[1][0] - i_bbox[0][0] + 1
        h_cell = i_bbox[1][1] - i_bbox[0][1] + 1
        x_cell = max(x_cell, 0)
        y_cell = max(y_cell, 0)
        w_cell = min(w_cell, w_img - x_cell)
        h_cell = min(h_cell, h_img - y_cell)
        cell_patch = img[y_cell:y_cell + h_cell, x_cell:x_cell + w_cell, :].copy()
        cnt_adj = (i_contour - np.array([x_cell, y_cell])).astype('int')
        mask = np.zeros(cell_patch.shape,dtype=np.int8)
        mask = cv2.drawContours(mask, [cnt_adj], -1, (1,1,1), cv2.FILLED)
        cell_patch[mask!=1] = 255
        pad_left = (patch_size - w_cell) // 2
        pad_right = patch_size - w_cell - pad_left
        pad_top = (patch_size - h_cell) // 2
        pad_bottom = patch_size - h_cell - pad_top
        pad_left=max(pad_left,0)
        pad_right=max(pad_right,0)
        pad_top=max(pad_top,0)
        pad_bottom=max(pad_bottom,0)
        cell_patch = np.pad(cell_patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=255)
        if cell_patch.shape != (patch_size, patch_size, 3): continue
        li_img_id.append(img_id+'.png')
        li_cell_id.append(img_id+'_{}_{}'.format(int(i_centroid[0]),int(i_centroid[1])))
        li_x.append(int(i_centroid[0]))
        li_y.append(int(i_centroid[1]))
        li_prop.append(i_prob)
        li_area.append(cv2.contourArea(cnt_adj))
        ellipse = cv2.fitEllipse(cnt_adj)
        li_len.append(max(ellipse[1]))
        li_cir.append(4*np.pi*cv2.contourArea(cnt_adj)/cv2.arcLength(cnt_adj,True)**2)
        li_sol.append(float(cv2.contourArea(cnt_adj))/cv2.contourArea(cv2.convexHull(cnt_adj)))
        pngs.append(cell_patch)
    if len(pngs)==0:continue
    np.save(os.path.join(core_dir,img_id+'.npy'),np.array(pngs))
    cell_csv=pd.DataFrame({'Image':li_img_id,'Name':li_cell_id,'Centroid X px':li_x,'Centroid Y px':li_y,'Detection probability':li_prop,'Area px^2':li_area,'Length px':li_len,'Circularity':li_cir,'Solidity':li_sol})
    cell_csv.to_csv(os.path.join(res_dir,'texts',img_id+'.csv'),index=False)




