medium_256 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 4,
    'max_times': 5,
    'max_width': 50,
    'max_angle': 4,
    'max_len': 100
    },
'box_proba': 0.3,
'box_kwargs':
    {
    'margin': 0,
    'bbox_min_size': 10,
    'bbox_max_size': 50,
    'max_times': 5,
    'min_times': 1,
    },
'segm_proba': 0,
'squares_proba': 0
}

medium_512 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 4,
    'max_times': 10,
    'max_width': 100,
    'max_angle': 4,
    'max_len': 200
    },
'box_proba': 0.3,
'box_kwargs':
    {
    'margin': 0,
    'bbox_min_size': 30,
    'bbox_max_size': 150,
    'max_times': 5,
    'min_times': 1,
    },
'segm_proba': 0,
'squares_proba': 0
}


thick_256 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 1,
    'max_times': 5,
    'max_width': 100,
    'max_angle': 4,
    'max_len': 200
    },
'box_proba': 0.3,
'box_kwargs':
    {
    'margin': 10,
    'bbox_min_size': 30,
    'bbox_max_size': 150,
    'max_times': 3,
    'min_times': 1,
    },
'segm_proba': 0,
'squares_proba': 0
}

thick_512 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 1,
    'max_times': 5,
    'max_width': 250,
    'max_angle': 4,
    'max_len': 450
    },
'box_proba': 0.3,
'box_kwargs':
    {
    'margin': 10,
    'bbox_min_size': 30,
    'bbox_max_size': 150,
    'max_times': 4,
    'min_times': 1,
    },
'segm_proba': 0,
'squares_proba': 0
}

thin_256 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 4,
    'max_times': 50,
    'max_width': 10,
    'max_angle': 4,
    'max_len': 40
    },
'box_proba': 0,
'segm_proba': 0,
'squares_proba': 0
}

thin_512 = {
'irregular_proba': 1,
'hole_range': [0.0,0.7],
'irregular_kwargs':
    {
    'min_times': 4,
    'max_times': 70,
    'max_width': 20,
    'max_angle': 4,
    'max_len': 100
    },
'box_proba': 0,
'segm_proba': 0,
'squares_proba': 0
}

segm_256 = {
"mask_gen_kwargs":
    {"confidence_threshold": 0.5},

"max_masks_per_image": 1,

"cropping":
    {"out_min_size": 256,
  "handle_small_mode": "upscale",
  "out_square_crop": True,
  "crop_min_overlap": 1},

"max_tamper_area": 0.5
}

segm_512 = {
"mask_gen_kwargs":
    {"confidence_threshold": 0.5},

"max_masks_per_image": 1,

"cropping":
    {"out_min_size": 512,
  "handle_small_mode": "upscale",
  "out_square_crop": True,
  "crop_min_overlap": 1},

"max_tamper_area": 0.5
}