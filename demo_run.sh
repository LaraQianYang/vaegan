python trainer_gan_h_s15_newloss.py \
    	--train_source_path=quora/100k/train_source.txt \
    	--train_target_path=quora/100k/train_target.txt \
    	--valid_source_path=quora/100k/test_source.txt \
    	--valid_target_path=quora/100k/test_target.txt \
    	--info_dir=quora/100k/info/ \
	--embed_size=300 \
       	--hidden_size=500 \
       	--latent_size=1000 \
	--number_highway_layers=2 \
	--number_rnn_layers=2 \
	--batch_size=5 \
	--test_batch_size=5 \
	--number_epochs=20 \
	--learning_rate=1e-4 \
	--dropout_rate=0.1 \
	--enforce_ratio=0.8 \
	--glove_path=glove/glove.6B.300d.txt \
	--valid_size=800 \
	--valid_every=3000 \
    	--save_path=SavedModels/nest-vae-test_quora_100k_newloss_3.ckpt \
	--train_print_every=100 \
	--valid_print_every=5000 \
	--alpha=0.8 \
	--recon_alpha=0.5 \
	--test=0 \
	--adv_epochs=100 \
	--adv_lr=1e-6 \
	--adv_ratio=5 \
	--grad_clip=10. \
	--adv_clip=5. \
	--use_baseline=1 \
	--gamma=1e-2 \
	--beta_gan=0. 

