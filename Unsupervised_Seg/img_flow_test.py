import numpy as np
import os, sys
import nrrd

'''
Flow from directory:
Author: Junyu Chen

Images must be 3D volumes
Inputs:
   img_dir:     image directory
   gt_dir:      ground truth directory
   file_type:   'nrrd' or 'npz'
   ndim:        '2D' or '3D'
   iter_dim:    which axis to be iterated
   
Outputs:
    img=[batch_size, sz_x, sz_y]
    gt=[batch_size, sz_x, sz_y]
'''

def img_read(img_loc,gt_loc,file_type):
    if file_type == 'nrrd':
        img, hdr = nrrd.read(img_loc)
        gt, hdr = nrrd.read(gt_loc)
    elif file_type == 'npz':
        img = np.load(img_loc)
        img = img['spect']
        gt = np.load(gt_loc)
        gt = gt['label']
    return img, gt

def find_arrow():
    with open(".img_flow_test.txt") as fp:
        line = fp.readline()
        line_num = 0
        while line:
            if "<-" in line:
                line_info  = line.split(',')
                img_name   = line_info[0]
                gt_name    = line_info[1]
                slice_info = line_info[2]
                slice_remain = line_info[3]
                line_arrow = line_num
            line_num += 1
            line = fp.readline()
    return img_name, gt_name, slice_info, slice_remain, line_arrow, line_num

def file_update(img_name,gt_name,slice_info,slice_remain,line_num):
    with open(".img_flow_test.txt", 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    #print(line_num)
    if slice_info.rstrip()=='' or slice_remain.rstrip()=='':
        data[line_num] = img_name.rstrip() + ',' + gt_name.rstrip() + '\n'
    else:
        data[line_num] = img_name.rstrip()+','+gt_name.rstrip()+','+slice_info.rstrip()+','+slice_remain.rstrip()+',<-\n'
    with open(".img_flow_test.txt", 'w') as file:
        file.writelines(data)

def get_line(line_num):
    with open(".img_flow_test.txt", 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    nextline = data[line_num]
    line_info = nextline.split(',')
    img_name = line_info[0]
    gt_name = line_info[1]
    return img_name, gt_name

def img_flow(img_dir, gt_dir, file_type, batch_size, ndim='2D', iter_dim = 'x', reset_flag=False):
    if reset_flag == True:
        os.remove(".img_flow_test.txt")
    if ndim == '3D':
        raise Exception('Only supports 2D for now!')
    img_files = sorted(os.listdir(img_dir))
    #print(img_files)
    gt_files = sorted(os.listdir(gt_dir))
    #print(gt_files)

    if not os.path.exists(".img_flow_test.txt"):
        outF = open(".img_flow_test.txt", "w+")
        for idx in range(len(img_files)):
            if idx == 0:
                outF.write(img_files[idx] + ',' + gt_files[idx]+',0:'+str(batch_size)+',0'+',<-')
            else:
                outF.write(img_files[idx] + ',' + gt_files[idx])
            outF.write("\n")
        outF.close()
    img_name, gt_name, slice_info, slice_remain, line_num, total_line = find_arrow()
    #print([img_name, gt_name, slice_info])

    img, gt = img_read(img_dir+img_name.rstrip(),gt_dir+gt_name.rstrip(),file_type)


    sz_x, sz_y, sz_z = img.shape
    slice_info = slice_info.split(':')
    nxt_slice = int(slice_info[1])+batch_size

    remain_slice = 0
    if iter_dim == 'x':
        # sort current img
        img_out = img[int(slice_info[0]):int(slice_info[1]),:,:]
        gt_out = gt[int(slice_info[0]):int(slice_info[1]),:,:]
        if int(slice_remain) != 0:
            if line_num == 0:
                pre_line = total_line-1
            else:
                pre_line = line_num - 1
            #print(total_line)
            img_name_pre, gt_name_pre = get_line(pre_line)
            img_pre, gt_pre = img_read(img_dir+img_name_pre.rstrip(),gt_dir+gt_name_pre.rstrip(),file_type)
            img_pre_out = img_pre[-1-int(slice_remain):-1,:,:]
            gt_pre_out  = gt_pre[-1 - int(slice_remain):-1, :, :]
            img_out = np.concatenate([img_out, img_pre_out])
            gt_out = np.concatenate([gt_out, gt_pre_out])
        # for next img
        if sz_x < nxt_slice:
            if line_num+1>total_line-1:
                nxt_line = 0
            else:
                nxt_line = int(line_num) + 1
            remain_slice = nxt_slice-sz_x
            slice_info = '0:' + str(remain_slice)
            img_name_nxt, gt_name_nxt = get_line(nxt_line)
            slice_remain = str(batch_size - remain_slice)
            file_update(img_name, gt_name, '', '', int(line_num))
            file_update(img_name_nxt, gt_name_nxt, slice_info, str(slice_remain), nxt_line)
        else:
            slice_remain = 0
            slice_info = slice_info[1] + ':' + str(nxt_slice)
            #print(slice_info)
            file_update(img_name, gt_name, slice_info, str(slice_remain), int(line_num))

    elif iter_dim == 'y':
        img_out = img[:,int(slice_info[0]):int(slice_info[1]),:]
        gt_out = gt[:,int(slice_info[0]):int(slice_info[1]),:]
        if sz_y < nxt_slice:
            remain_slice = nxt_slice-sz_y
    elif iter_dim == 'z':
        img_out = img[:, :, int(slice_info[0]):int(slice_info[1])]
        gt_out = gt[:, :, int(slice_info[0]):int(slice_info[1])]
        if sz_y < nxt_slice:
            remain_slice = nxt_slice-sz_z

    #print(img_out.shape)
    return img_out, gt_out


#img_dir = '/netscratch/jchen/gary_phan/imgs/'
#gt_dir  = '/netscratch/jchen/gary_phan/gt/'
#for i in range(100):
#    img_flow(img_dir, gt_dir, 'nrrd', 100, '2D','x', False)