#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.models.mlp import MLP
from mmbt.models.bert import BertClf
from mmbt.models.bow import GloveBowClf
from mmbt.models.concat_bert import MultimodalConcatBertClf
from mmbt.models.concat_bow import  MultimodalConcatBowClf, MultimodalConcatBow16Clf, MLPGenreClf
from mmbt.models.image import ImageClf
from mmbt.models.mmbt import MultimodalBertClf, MultimodalBertEncoder2M
from mmbt.models.mmbtadapter import MultimodalBertAdapterClf
from mmbt.models.mmbtadapterm import MultimodalBertAdapterMClf, MultimodalBertAdapterMTropesClf
from mmbt.models.mmbt3 import MultimodalBertThreeClf
from mmbt.models.gmu import GMUClf
from mmbt.models.mmtr import (MMTransformerClf, SimpleGMUClf, TranslatingMMTransformerGMUClf_TPrepro,
                            MMTransformerGMUClf, MultiprojectionMMTransformerGMUClf, MultiprojectionMMTransformer3DGMUClf,
                            MMTransformerGMUHybridClf,
                            MMTransformerGMUClfVAPT,
                            TranslatingMMTransformerGMUClf,
                            TranslatingMMTransformerGMUClf_residual_v3,
                            TranslatingMMTransformerGMUClf_residual_v4T,
                            TranslatingMMTransformerGMUClf_residual_v4_hybrid,
                            TranslatingMMTransformerGMUClf_early_fusion,
                            TranslatingMMTransformerGMUClf_residual_v4,
                            TranslatingMMTransformerMAGClf,
                            MMTransformerUniClf,
                            TransformerClf,
                            MMTransformerUniBi,
                            TransformerVideoClf,
                            TransformerAudioClf,
                            MMTransformerMoviescopeClf,
                            MMTransformerGMUMoviescopeVidTextClf,
                            MMTransformerGMUMoviescopeVidAudClf,
                            MMTransformerConcatMoviescopeVidAudClf,
                            MMTransformerGMUMoviescopeTxtAudClf,
                            MMTransformerConcatMoviescopeTxtAudClf,
                            MMTransformerGMUMoviescopeVidAudPosterClf,
                            MMTransformerConcatMoviescopeVidAudPosterClf,
                            MMTransformerGMUMoviescopeVidAudPosterTxtClf,
                            MMTransformerConcatMoviescopeVidAudPosterTxtClf,
                            MMTransformerGMUMoviescopeClf,
                            MMTransformerConcatMoviescopeClf,
                            MMTransformerGMUVPAClf,
                            MMTransformerConcatVPAClf,
                            MMTransformerGMUNoEncodersClf,
                            MMTransformerGMU5NoEncodersClf,
                            MMTransformerGMU3BlockNoEncodersClf,
                            MMTransformerGMUHierarchicalNoEncodersClf,
                            MMTransformerGMUHierarchical3BlocksNoEncodersClf,
                            MMTransformerGMU4MoviescopeClf,
                            MMTransformerConcat4MoviescopeClf,
                            MMTransformerConcat5MoviescopeClf,
                            MMTransformerGMU5MoviescopeClf,
                            MMTransformerGMU5IntraMoviescopeClf,
                            MMTransformerHierarchicalEncoderOnlyMoviescopeClf,
                            MMTransformerHierarchicalFullMoviescopeClf)

MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbow16": MLPGenreClf,
    "gmu": GMUClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmbtadapter": MultimodalBertAdapterClf,
    "mmbtadapterm": MultimodalBertAdapterMClf,
    "mmbt3": MultimodalBertThreeClf,
    "mmtr": TransformerVideoClf,
    "mmtra": TransformerAudioClf,
    "mmtrv": MMTransformerMoviescopeClf, # text-video (MMTransformerGMUMoviescopeVidTextClf)
    "mmtrva": MMTransformerConcatMoviescopeVidAudClf, # video-audio (no text) (MMTransformerGMUMoviescopeVidAudClf)
    "mmtrta": MMTransformerGMUMoviescopeTxtAudClf, # plot-audio (MMTransformerConcatMoviescopeTxtAudClf)
    "mmtrvap": MMTransformerConcatMoviescopeVidAudPosterClf, # video-audio-poster (no text) (MMTransformerGMUMoviescopeVidAudPosterClf)
    "mmtrvapt": MultiprojectionMMTransformerGMUClf, #TranslatingMMTransformerGMUClf_TPrepro, #SimpleGMUClf, #TranslatingMMTransformerMAGClf,#TranslatingMMTransformerGMUClf_residual_v4, # TranslatingMMTransformerGMUClf_early_fusion, # TranslatingMMTransformerGMUClf_residual_v4_hybrid, #TranslatingMMTransformerGMUClf_residual_v4T, #MMTransformerGMUClfVAPT,# TranslatingMMTransformerGMUClf_residual_v3, # video-audio-poster-plot (MMTransformerGMUHybridClf, MMTransformerGMUMoviescopeVidAudPosterTxtClf)
    "mmtrvat": MultiprojectionMMTransformer3DGMUClf,#TranslatingMMTransformerGMUClf,
    "mmtrvpp": MMTransformerGMUMoviescopeClf, # video-plot-poster (MMTransformerGMUMoviescopeClf)
    "mmtrvpa": MMTransformerGMUNoEncodersClf, # video-plot-audio (MMTransformerGMUVPAClf, MMTransformerConcatVPAClf, MMTransformerGMUNoEncodersClf, MMTransformerHierarchicalEncoderOnlyMoviescopeClf, MMTransformerHierarchicalFullMoviescopeClf, MMTransformerGMUHierarchicalNoEncodersClf, MMTransformerGMUHierarchical3BlocksNoEncodersClf, MMTransformerGMU3BlockNoEncodersClf)
    "mmtrvppm": MMTransformerConcat4MoviescopeClf, # video-plot-poster-metadata (MMTransformerConcat4MoviescopeClf)
    "mmtrvpapm": MMTransformerGMU5NoEncodersClf, # video-plot-audio-poster-metadata (MMTransformerGMU5MoviescopeClf, MMTransformerConcat5MoviescopeClf, MMTransformerGMU5IntraMoviescopeClf, MMTransformerGMU5NoEncodersClf)
    "mmbtp": MultimodalBertTransfClf,
    "mmdbt": MultimodalDistilBertClf,
    "vilbert": VILBertForVLTasks,
    "mmbtrating": MultimodalBertRatingClf,
    "mmbtratingtext": MultimodalBertRatingTextClf,
    "mmtrrating": MMTransformerRatingClf,
}


def get_model(args, config=None):
    # Initialize ViLBERT model
    if config:
        config.args = args
        return VILBertForVLTasks.from_pretrained(args.from_pretrained,
                                                 config=config)
    # Initialize all other models
    return MODELS[args.model](args)
