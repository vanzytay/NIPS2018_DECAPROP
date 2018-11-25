from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

def build_parser():
    """ I use the same growing list of arguments for all repositories.
    So you might find things that are not used here.
    """
    parser = argparse.ArgumentParser()
    ps = parser.add_argument
    ps("--dataset", dest="dataset", type=str,  default='TrecQA', help="(WikiQA)")
    ps('--features',type=int, metavar='<int>',
       default=0, help='Used features or not')
    ps("--rnn_type", dest="rnn_type", type=str, metavar='<str>',
        default='DECAPROP_SM_RP',
       help="controls the model type")
    ps("--comp_layer", dest="comp_layer", type=str, metavar='<str>',
        default='MUL_SUB',help="")
    ps("--mca_types", dest="mca_types", type=str, metavar='<str>',
       default='ALIGN_MAX_MEAN_INTRA',help="type of coattention")
    ps("--fm_comp_layer", dest="fm_comp_layer", type=str, metavar='<str>',
        default='MUL_SUB_CAT', help="type of FM function")
    ps("--add_features", dest="add_features", type=str, metavar='<str>',
        default='', help="add_features into representations")
    ps("--ans_features", dest="ans_features", type=str, metavar='<str>',
        default='', help="add ans features into representations")
    ps("--char_enc", dest="char_enc", type=str, metavar='<str>',
        default='CNN',help="type of char encoder)")
    ps("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
       help="Optimization algorithm)")
    ps("--emb_size", dest="emb_size", type=int, metavar='<int>',
       default=300, help="Embeddings dimension (default=300)")
    ps("--story_emb_size", dest="story_emb_size", type=int, metavar='<int>',
       default=100, help="Story Embeddings dimension (default=100)")
    ps("--int_emb_size", dest="int_emb_size", type=int, metavar='<int>',
      default=15, help="Latent factors for interaction embeddings")
    ps("--char_emb_size", dest="char_emb_size", type=int, metavar='<int>',
       default=20, help="Embeddings dimension")
    ps("--qt_emb_size", dest="qt_emb_size", type=int, metavar='<int>',
       default=8, help="Embeddings dimension")
    ps("--pos_emb_size", dest="pos_emb_size", type=int, metavar='<int>',
       default=20, help="Embedding Dimension")
    ps("--rnn_size", dest="rnn_size", type=int, metavar='<int>',
       default=64, help="RNN dimension. (default=300)")
    ps("--proj_size", dest="proj_size", type=int, metavar='<int>',
       default=300, help="Size of projection")
    ps("--cnn_size", dest="cnn_size", type=int, metavar='<int>',
       default=100, help="CNN dimension. (default=300)")
    ps("--conv_size", dest="conv_size", type=int, metavar='<int>',
       default=5, help="Conv Size for QRNN (default=3)")
    ps("--align_spans", dest="align_spans", type=int, metavar='<int>',
       default=0, help="Use align spans or not..")
    ps("--use_lower", dest="use_lower", type=int, metavar='<int>',
       default=0, help="Use all lowercase")
    ps("--batch-size", dest="batch_size", type=int, metavar='<int>',
       default=100, help="Batch size (default=256)")
    ps("--allow_growth", dest="allow_growth", type=int, metavar='<int>',
      default=0, help="Allow Growth")
    ps("--rnn_layers", dest="rnn_layers", type=int,
       metavar='<int>', default=3, help="Number of RNN layers")
    ps("--decay_epoch", dest="decay_epoch", type=int,
       metavar='<int>', default=0, help="Decay everywhere n epochs")
    ps("--num_dense", dest="num_dense", type=int,
       metavar='<int>', default=0, help="Number of dense layers")
    ps("--num_proj", dest="num_proj", type=int, metavar='<int>',
       default=1, help="Number of projection layers")
    ps("--hierarchical", dest="hierarchical", type=int, metavar='<int>',
       default=0, help="hierarchical mode")
    ps("--rnn_constant", dest="rnn_constant", type=int, metavar='<int>',
       default=0, help="Fix Rnn size to same as input")
    ps("--clip_output", dest="clip_output", type=int, metavar='<int>',
        default=0, help="clip output")
    ps("--proj_intra", dest="proj_intra", type=int, metavar='<int>',
       default=0, help="Project intra embeddings to original size")
    ps("--normalize_embed", dest="normalize_embed", type=int, metavar='<int>',
      default=0, help="Normalize pretrained embeddings")
    ps("--factor", dest="factor", type=int, metavar='<int>',
       default=32, help="For factorization factors")
    ps("--factor2", dest="factor2", type=int, metavar='<int>',
       default=32, help="Inner factorization features")
    ps("--rnn_direction", dest="rnn_direction", type=str,
       metavar='<str>', default='uni', help="Direction of RNN")
    ps("--aggregation", dest="aggregation", type=str, metavar='<str>',
        default='last', help="The aggregation method for regp and bregp types")
    ps("--dropout", dest="dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--rnn_dropout", dest="rnn_dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--aggr_layers", dest="aggr_layers", type=int,
       metavar='<int>', default=1, help="Number of aggregation layers")
    ps("--emb_dropout", dest="emb_dropout", type=float, metavar='<float>',
        default=1.0, help="The dropout probability.")
    ps("--pretrained", dest="pretrained", type=int, metavar='<int>',
       default=1, help="Whether to use pretrained or not")
    ps("--adjust", dest="adjust", type=int, metavar='<int>',
       default=0, help="To adjust or not")
    ps("--patience", dest="patience", type=int, metavar='<int>',
       default=3, help="Patience for halving LR")
    ps("--use_topk", dest="use_topk", type=int, metavar='<int>',
       default=1, help="To use topk or not")
    ps("--epochs", dest="epochs", type=int, metavar='<int>',
       default=10, help="Number of epochs (default=50)")
    ps("--hogger", dest="hogger", type=int, metavar='<int>',
       default=1, help="To hog GPU or not")
    ps("--choice_mask", dest="choice_mask", type=int, metavar='<int>',
       default=0, help="To apply choice mask")
    ps("--shuffle", dest="shuffle", type=int, metavar='<int>',
       default=1, help="To train shuffle or not")
    ps("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
       help="Maximum allowed number of words during training. '0' \
            means no limit (default=0)")
    ps('--gpu', dest='gpu', type=str, metavar='<str>',
       default='0', help="Specify which GPU to use (default=0)")
    ps('--add_questions', dest='add_questions', type=int, metavar='<int>',
       default='0', help="Add questions to answers")
    ps("--gpu_hog", dest='gpu_hog', type=int, metavar='<int>',
       default=1, help="to hog all gpu space or not")
    ps("--hdim", dest='hdim', type=int, metavar='<int>',
       default=300, help="Hidden layer size (default=50)")
    ps("--lr", dest='lr', type=float,
       metavar='<float>', default=0.5, help="Learning Rate")
    ps("--dev_lr", dest='dev_lr', type=int,
       metavar='<int>', default=0, help="Dev Learning Rate")
    ps("--margin", dest='margin', type=float,
       metavar='<float>', default=0.2, help="Margin")
    ps("--clip_norm", dest='clip_norm', type=int,
       metavar='<int>', default=5, help="Clip Norm value")
    ps("--trainable", dest='trainable', type=int, metavar='<int>',
       default=0, help="Trainable Word Embeddings (0|1)")
    ps('--l2_reg', dest='l2_reg', type=float, metavar='<float>',
       default=0.0, help='L2 regularization, default=4E-6')
    ps('--eval', dest='eval', type=int, metavar='<int>',
       default=1, help='Epoch to evaluate results')
    ps('--log', dest='log', type=int, metavar='<int>',
       default=1, help='1 to output to file and 0 otherwise')
    ps('--local_feats', dest='local_feats', type=int, metavar='<int>',
       default=0, help='To use local features or not')
    ps('--use_pos', dest='use_pos', type=int, metavar='<int>',
       default=0, help='To use pos features or not')
    ps('--flow_inner', dest='flow_inner', type=int, metavar='<int>',
       default=0, help='To flow inner att or not')
    ps('--flow_intra', dest='flow_intra', type=int, metavar='<int>',
       default=1, help='To flow intra att or not')
    ps('--enc_only', dest='enc_only', type=int, metavar='<int>',
       default=0, help='To use encoder only and no cross sent att')
    ps('--dev', dest='dev', type=int, metavar='<int>',
       default=1, help='1 for development set 0 to train-all')
    ps('--seed', dest='seed', type=int, default=1337, help='random seed')
    ps('--num_heads', dest='num_heads', type=int, default=1,
        help='number of heads')
    ps('--num_hr', dest='num_hr', type=int, default=2,
        help='number of hierarchical reasoning iters')
    ps('--story_window', dest='story_window', type=int, default=3,
        help='story_window')
    ps("--vocab", dest="vocab", type=int, metavar='<int>',
       default=0, help="Vocab limit (set 0 to no limit)")
    ps("--hard", dest="hard", type=int, metavar='<int>',
       default=1, help="Use hard att when using gumbel")
    ps('--toy', action='store_true', help='Use toy dataset (for fast testing)')
    ps('--tensorboard', action='store_true', help='To use tensorboard or not')
    ps('--early_stop',  dest='early_stop', type=int,
       metavar='<int>', default=3, help='early stopping')
    ps('--wiggle_lr',  dest='wiggle_lr', type=float,
       metavar='<float>', default=1E-5, help='Wiggle lr')
    ps('--wiggle_after',  dest='wiggle_after', type=int,
       metavar='<int>', default=0, help='Wiggle lr')
    ps('--wiggle_score',  dest='wiggle_score', type=float,
       metavar='<float>', default=0.0, help='Wiggle score')
    ps('--translate_proj', dest='translate_proj', type=int,
       metavar='<int>', default=0, help='To translate project or not')
    ps('--test_bsz', dest='test_bsz', type=int,
       metavar='<int>', default=4, help='Multiplier for eval bsz')
    ps('--eval_train', dest='eval_train', type=int,
       metavar='<int>', default=1, help='To eval on train set or not')
    ps('--final_layer', dest='final_layer', type=int,
       metavar='<int>', default=1, help='To use final layer or not')
    ps('--padding', dest='padding', type=str, default='post',
        help='pre or post')
    ps('--data_link', dest='data_link', type=str, default='',
        help='data link')
    ps('--att_type', dest='att_type', type=str, default='SOFT',
        help='attention type')
    ps('--att_pool', dest='att_pool', type=str, default='MAX',
        help='pooling type for attention')
    ps('--proj_mode', dest='proj_mode', type=str, default='FC',
        help='projection of embedding layer')
    ps('--num_class', dest='num_class', type=int,
       default=2, help='self explainatory..')
    ps('--predict_binary', dest='predict_binary', type=int,
       default=0, help='Predict binary for ranking with binary metric')
    ps('--num_extra_features', dest='num_extra_features',
       type=int, default=4, help='4 features for TREC QA')
    ps('--test_eval', dest='test_eval',
       type=int, default=1, help='To test eval or not')
    ps('--all_dropout', action='store_true',
       default=False, help='softmax similarity layer')
    ps("--qmax", dest="qmax", type=int, metavar='<int>',
       default=25, help="Max Length of Question")
    ps("--char_max", dest="char_max", type=int, metavar='<int>',
       default=8, help="Max length of characters")
    ps("--amax", dest="amax", type=int, metavar='<int>',
       default=15, help="Max Length for Answer")
    ps("--smax", dest="smax", type=int, metavar='<int>',
       default=200, help="Max Length of Sentences")
    ps("--pmax", dest="pmax", type=int, metavar='<int>',
       default=500, help="Max Length of passages (baidu)")
    ps("--paramax", dest="paramax", type=int, metavar='<int>',
       default=50, help="Max Length of paragraph (baidu)")
    ps("--test_bsz_div", dest="test_bsz_div", type=int, metavar='<int>',
       default=1, help="Test Bsz divider")
    ps("--paramax_first", dest="paramax_first", type=int, metavar='<int>',
       default=150, help="Max Length of highest priority paragraph")
    ps("--tmax", dest="tmax", type=int, metavar='<int>',
       default=1000, help="Max Length of Span-based Tests")
    ps("--dmax", dest="dmax", type=int, metavar='<int>',
       default=200, help="Max Number of documents")
    ps("--burn", dest="burn", type=int, metavar='<int>',
       default=0, help="Burn in period..")
    ps("--var_drop", dest="var_drop", type=int, metavar='<int>',
       default=1, help="Use variational dropout")
    ps("--num_neg", dest="num_neg", type=int, metavar='<int>',
       default=6, help="Number of negative samples")
    ps("--injection", dest="injection", type=int, metavar='<int>',
       default=0, help="For hyperparameter injection")
    ps('--constraint',  type=int, metavar='<int>',
       default=0, help='Constraint embeddings to unit ball')
    ps('--sampling_mode', dest='sampling_mode',
       default='Mix', help='Which sampling mode..')
    ps('--base_encoder', dest='base_encoder',
       default='LAST_LSTM', help='BaseEncoder for H-mode')
    ps('--save_pred', action='store_true', default=False,
       help='Save predictions for ensemble')
    ps('--save_embed', action='store_true', default=False,
       help='Save embeddings for visualisation')
    ps('--ensemble', action='store_true', default=False, help='Use ensemble')
    ps('--default_len', dest="default_len", type=int, metavar='<int>',
       default=1, help="Use default len or not")
    ps('--sort_batch', dest="sort_batch", type=int, metavar='<int>',
       default=0, help="To use sort-batch optimization or not")
    ps('--factor_proj', dest="factor_proj", type=int, metavar='<int>',
       default=0, help="To use factor_proj or not")
    ps("--init", dest="init", type=float,
       metavar='<float>', default=0.01, help="Init Params")
    ps("--ema_decay", dest="ema_decay", type=float,
       metavar='<float>', default=0, help="Ema decay..")
    ps("--temperature", dest="temperature", type=float,
      metavar='<float>', default=0.5, help="Temperature")
    ps("--num_intra_proj", dest="num_intra_proj", type=int,
       metavar='<int>', default=1, help="Number of intra projection layers")
    ps("--num_ap_proj", dest="num_ap_proj", type=int,
       metavar='<int>', default=1, help="Number of AP projection layers")
    ps("--num_inter_proj", dest="num_inter_proj", type=int,
       metavar='<int>', default=1, help="Number of inter projection layers")
    ps("--dist_bias", dest="dist_bias", type=int,
       metavar='<int>', default=0, help="To use distance bias for intra-att or not")
    ps("--num_com", dest="num_com", type=int,
       metavar='<int>', default=1, help="Number of compare layers")
    ps("--num_cocoa", dest="num_cocoa", type=int,
      metavar='<int>', default=1, help="Number of cocoa layers")
    ps("--show_att", dest="show_att", type=int,
      metavar='<int>', default=0, help="Display Attention")
    ps("--write_qual", dest="write_qual", type=int,
    metavar='<int>', default=0, help="write qual")
    ps("--show_affinity", dest="show_affinity", type=int,
    metavar='<int>', default=0, help="Display Affinity Matrix")
    ps("--append_question", dest="append_question", type=int,
        metavar='<int>', default=0, help="Append question to answer")
    ps("--max_span", dest="max_span", type=int,
        metavar='<int>', default=15, help="max span")
    ps("--init_type", dest="init_type", type=str,
       metavar='<str>', default='xavier', help="Init Type")
    ps("--rnn_init_type", dest="rnn_init_type", type=str,
       metavar='<str>', default='xavier', help="RNN Init Type")
    ps("--rnn_init", dest="rnn_init", type=float,
       metavar='<float>', default='0.1', help="RNN Init val")
    ps("--choice_layer", dest="choice_layer", type=str,
      metavar='<str>', default='BILINEAR', help="type of choice layer")
    ps("--dq_pooling", dest="dq_pooling", type=str,
      metavar='<str>', default='MAX', help="Pooling for document-query coatt")
    ps("--dq_att", dest="dq_att", type=str,
      metavar='<str>', default='BILINEAR', help="Sim func. for document-query coatt")
    ps("--agr_att", dest="agr_att", type=str,
      metavar='<str>', default='SOFT', help="Sim func. for self-att aggr coatt")
    ps("--da_pooling", dest="da_pooling", type=str,
      metavar='<str>', default='MAX', help="Pooling for document-choice coatt")
    ps("--init_emb", dest="init_emb", type=float,
       metavar='<float>', default=0.01, help="Init Embeddings")
    ps("--decay_lr", dest="decay_lr", type=float,
       metavar='<float>', default=0, help="Decay Learning Rate")
    ps("--decay_steps", dest="decay_steps", type=float,
       metavar='<float>', default=0, help="Decay Steps (manual)")
    ps("--decay_stairs", dest="decay_stairs", type=float,
       metavar='<float>', default=1, help="To use staircase or not")
    ps('--supply_neg', action='store_true', default=False,
       help='Supply neg samples to training set each iter')
    ps('--reward_func', dest='reward_func', type=str,
       default='mixed_bleu', help='reward func')
    ps('--emb_type', dest='emb_type', type=str,
       default='glove', help='embedding type')
    ps('--swap_mode', dest='swap_mode', type=str,
       default='none', help='swap_mode')
    ps('--log_dir', dest='log_dir', type=str,
       default='logs', help='log directory')
    ps("--num_choice", dest="num_choice", type=int, metavar='<int>',
       default=4, help="Max number of choices")
    ps("--num_gpu", dest="num_gpu", type=int, metavar='<int>',
       default=1, help="Number of gpus (experimental)")
    ps("--start_pos", dest="start_pos", type=int, metavar='<int>',
       default=200, help="Start Position for Swap")
    ps("--test_chunks", dest="test_chunks", type=int, metavar='<int>',
       default=200, help="Size of test chunks")
    ps('--fren_heads', dest='fren_heads', type=int, default=4,
        help='number of fren heads')
    ps('--fren_mems', dest='fren_mems', type=int, default=64,
        help='number of fren mems')
    ps("--hmode", dest="hmode", type=int, metavar='<int>',
       default=0, help="Whether to use hmode or not")
    ps("--train_rnn_init", dest="train_rnn_init", type=int, metavar='<int>',
       default=1, help="Whether to train RNN init vector or not")
    ps("--mask_softmax", dest='mask_softmax', type=int,
        metavar='<int>', default=0, help="To use mask softmax or not")
    ps('--save', default=1, type=int,
        metavar='<int>',help='To save model or not?')
    ps("--edrop", dest="edrop", type=float, metavar='<float>',
        default=0.8,help="Embedding dropout.")
    ps("--hard_ratio", dest="hard_ratio", type=float, metavar='<float>',
        default=0.2,help="Ratio for hard set.")
    ps("--rl_ratio", dest="rl_ratio", type=float, metavar='<float>',
        default=0.5, help="Ratio for RL loss.")
    ps("--hdrop", dest="hdrop", type=float,
        metavar='<float>', default=0.8,
        help="Hierarchical dropout probability.")
    ps("--cell_dropout", dest="cell_dropout", type=float, metavar='<float>',
        default=1.0, help="The dropout probability.")
    ps('--use_cudnn', dest='use_cudnn', type=int, default=1)
    ps('--use_cove', dest='use_cove', type=int, default=0)
    ps('--beam_size', dest='beam_size', type=int, default=1,
            help='beam_size'))
    ps('--mask_entities', dest='mask_entities', type=int, default=0,
            help='use entity masking..')
    ps('--use_elmo', dest='use_elmo', type=int, default=0)
    ps('--use_pg', dest='use_pg', type=int, default=1,
            help='use pointer generator (default=1)')
    ps('--use_fq', dest='use_fq', type=int, default=0,
            help='use document frequency')
    ps('--decoder_size', dest='decoder_size', type=int, default=512,
            help='size of the decoder')
    ps('--teacher_forcing', dest='teacher_forcing', type=int, default=1,
            help='to use teacher forcing or not')
    return parser
