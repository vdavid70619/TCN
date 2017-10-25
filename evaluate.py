import os
import cPickle as pickle
import json

os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

from utils import gen_json, nms_all


if __name__ == '__main__':

    FEATURE = os.environ['ACTNET_HOME'] + '/tsn_score/'

    with open('actNet200-V1-3.pkl', 'rb') as f:
        gt = pickle.load(f)

    # rank proposal
    with open('test_proposals.pkl', 'rb') as f:
        test_proposals = pickle.load(f)
    import train_proposal_pair2
    ranked_proposals = train_proposal_pair2.rank_propsal_pairs(gt['database'], test_proposals, model='rank_pair2_iter_1000.caffemodel', feature_path=FEATURE)
    with open("ranked_test_proposals.pkl", "wb") as f:
        pickle.dump(ranked_proposals, f)


    # test on video
    import train_proposal_class2
    score_tsn = train_proposal_class2.classify_video(gt['database'], subset='testing', feature_path=FEATURE)
    import train_chunk_mbh
    score_mbh = train_chunk_mbh.classify_video(model='chunk_mbh_iter_1000.caffemodel' , subset='testing', FEATURE=os.environ['ACTNET_HOME']+'/mbh/')
    import train_chunk_imshuffle
    score_ims = train_chunk_imshuffle.classify_video(model='chunk_ims_iter_1000.caffemodel' , subset='testing', FEATURE=os.environ['ACTNET_HOME']+'/shuffle_imagenet/')


    ranked_proposals =  nms_all(ranked_proposals, topK=20, nms_thor=0.45)
    classified_proposals = train_proposal_class2.classify_proposal(gt['database'], ranked_proposals, model='class_bl2_iter_1000.caffemodel',
                                             priors=[score_tsn, score_mbh, score_ims], weights=[1,1,1],
                                             feature_path=FEATURE, topK=1)
    classified_proposals = nms_all(classified_proposals, topK=20, nms_thor=1, remove_background=True)

    # test on offical evaluation code
    names = gt['actionIDs']
    id2name = {}
    for name, ids in names.iteritems():
        id2name[ids['class']] = name
    output = gen_json(classified_proposals, id2name)
    with open('prediction.json', 'w') as f:
        json.dump(output, f)