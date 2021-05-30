import argparse
from moco import MoCoMethodParams
from moco import MoCoMethod
from linear_classifier import LinearClassifierMethod

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-a', '--arch', default='resnet50')
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('-m', '--method', default='moco', type=str)
parser.add_argument('-d', '--dataset-name', default='stl10', type=str)
parser.add_argument('--id-weight', default=0, type=float)  # Identity Embedding Weight
parser.add_argument('--id-type', default=None, type=str)  # Identity Embedding Type


def getMethodParam(method):
    if method == 'moco':
        hparams = MoCoMethodParams()

    elif method == 'byol':
        hparams = MoCoMethodParams(
            prediction_mlp_layers=2,
            mlp_normalization="bn",
            loss_type="ip",
            use_negative_examples_from_queue=False,
            use_both_augmentations_as_queries=True,
            use_momentum_schedule=True,
            optimizer_name="lars",
            exclude_matching_parameters_from_lars=[".bias", ".bn"],
            loss_constant_factor=2
        )

    elif method == 'simclr':
        hparams = MoCoMethodParams(
            use_negative_examples_from_batch=True,
            use_negative_examples_from_queue=False,
            K=0,
            m=0.0,
            use_both_augmentations_as_queries=True,
        )

    elif method == 'eqco':
        hparams = MoCoMethodParams(use_eqco_margin=True, eqco_alpha=65536, K=256)

    else:
        assert False, f'{method} not foune!'

    return hparams


if __name__ == '__main__':
    from attr import evolve
    import os
    import pytorch_lightning as pl

    args = parser.parse_args()
    print(args)

    hparams = getMethodParam(args.method)
    hparams = evolve(hparams,
                     encoder_arch=args.arch,
                     batch_size=args.batch_size,
                     dataset_name=args.dataset_name,
                     id_weight=args.id_weight,
                     method=args.method)
    print(hparams)

    os.environ["DATA_PATH"] = "./data"

    """
    train
    """
    model = MoCoMethod(hparams)
    trainer = pl.Trainer(gpus=1, max_epochs=320)
    trainer.fit(model)

    ckpt_path = f'./checkpoints/{args.method}-{args.dataset_name}.pth.tar'
    if args.id_weight > 0:
        ckpt_path += f'{int(args.id_weight * 100):03d}-{args.id_type}'
    trainer.save_checkpoint(ckpt_path)

    # """
    # test
    # """
    # linear_model = LinearClassifierMethod.from_moco_checkpoint(ckpt_path,
    #                                                            dataset_name="stl10")
    # trainer = pl.Trainer(gpus=1, max_epochs=1)
    # trainer.fit(linear_model)