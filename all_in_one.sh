
# define home dir, should contain tsn_score, mbh, shuffle_imagenet features path
export PYTHONPATH=$PYTHONPATH:path
export ACTNET_HOME=path

python -u generate_proposal.py

python -u train_proposal_pair2.py
python -u train_proposal_class2.py

python -u train_chunk_mbh.py
python -u train_chunk_imshuffle.py

python -u evaluate.py
