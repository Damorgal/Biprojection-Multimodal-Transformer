#!/usr/bin/env python3

from bpmult.models.mmtr import(  MultiprojectionMMTransformerGMUClf,
                            MultiprojectionMMTransformer3DGMUClf)

MODELS = {
    "mmtrvapt": MultiprojectionMMTransformerGMUClf, #TranslatingMMTransformerGMUClf_TPrepro, #SimpleGMUClf, #TranslatingMMTransformerMAGClf,#TranslatingMMTransformerGMUClf_residual_v4, # TranslatingMMTransformerGMUClf_early_fusion, # TranslatingMMTransformerGMUClf_residual_v4_hybrid, #TranslatingMMTransformerGMUClf_residual_v4T, #MMTransformerGMUClfVAPT,# TranslatingMMTransformerGMUClf_residual_v3, # video-audio-poster-plot (MMTransformerGMUHybridClf, MMTransformerGMUMoviescopeVidAudPosterTxtClf)
    "mmtrvat": MultiprojectionMMTransformer3DGMUClf,#TranslatingMMTransformerGMUClf,
}


def get_model(args, config=None):
    # Initialize models
    return MODELS[args.model](args)
