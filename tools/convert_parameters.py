import argparse

import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, required=True,
    )
    # parser.add_argument(
    #     '--save_path', type=str, required=True,
    # )
    parser.add_argument(
        '--dataset', type=str, default='hico',
    )
    parser.add_argument(
        '--num_queries', type=int, default=100,
    )

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path)

    obj_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    # For no pair
    obj_ids.append(21)

    for k in list(ps['model'].keys()):
        print(k)
        if len(k.split('.')) > 1 and k.split('.')[1] == 'decoder':
            ps['model'][k.replace('decoder', 'instance_decoder')] = ps['model'][k].clone()
            ps['model'][k.replace('decoder', 'interaction_decoder')] = ps['model'][k].clone()
            del ps['model'][k]

    # ps['model']['hum_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
    # ps['model']['hum_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
    # ps['model']['hum_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

    # ps['model']['obj_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
    # ps['model']['obj_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
    # ps['model']['obj_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

    # ps['model']['obj_class_embed.weight'] = ps['model']['class_embed.weight'].clone()[obj_ids]
    # ps['model']['obj_class_embed.bias'] = ps['model']['class_embed.bias'].clone()[obj_ids]

    # ps['model']['query_embed_h.weight'] = ps['model']['query_embed_h.weight'].clone().repeat(2, 1)[:args.num_queries]
    # ps['model']['query_embed_o.weight'] = ps['model']['query_embed_o.weight'].clone().repeat(2, 1)[:args.num_queries]
    # ps['model']['pos_guided_embedd.weight'] = ps['model']['pos_guided_embedd.weight'].clone().repeat(2, 1)[:args.num_queries]
    # ps['model']['visual_projection.weight'] = ps['model']['visual_projection.weight'].clone()[:args.num_queries,:]
    # ps['model']['visual_projection.bias'] = ps['model']['visual_projection.bias'].clone()[:args.num_queries]
    # ps['model']['obj_visual_projection.weight'] = ps['model']['visual_projection.weight'].clone()[:16]
    # ps['model']['obj_visual_projection.bias'] = ps['model']['visual_projection.bias'].clone()[:16]
    print(ps['model']['obj_visual_projection.weight'].shape)
    print(ps['model']['obj_visual_projection.bias'].shape)
    if args.dataset == 'vcoco':
        l = nn.Linear(ps['model']['obj_class_embed.weight'].shape[1], 1)
        l.to(ps['model']['obj_class_embed.weight'].device)
        ps['model']['obj_class_embed.weight'] = torch.cat((
            ps['model']['obj_class_embed.weight'][:-1], l.weight, ps['model']['obj_class_embed.weight'][[-1]]))
        ps['model']['obj_class_embed.bias'] = torch.cat(
            (ps['model']['obj_class_embed.bias'][:-1], l.bias, ps['model']['obj_class_embed.bias'][[-1]]))

    #torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
