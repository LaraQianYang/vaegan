import argparse
import time
import torch
import torch.optim as optim

from modules_gan_newloss import *
from utils_gan_newloss import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# 02/13, new rouge score
#import rouge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_batch(model, optimizer, batch, indexer, 
              step, recon_alpha=0.5, training=True):
    """Run a batch of paraphrase inputs.
    
    Args:
        model: NestedVAE object.
        optimizer: torch.optim.Adam.
        batch: (source, target) tuple.
               source: torch LongTensor, shape = <batch-size, seq-length>.
               target: same as `source`, with a different seq-length.
        indexer: Indexer object.
        step: train/valid step index.
        training: model select boolean.
        print_result: print out prediction-true if True.
    Returns:
        torch Tensor unit loss.
    """
    # Run model.
    source, target = batch
    logits_t, kl_loss = model(source, target, training)
    logits, logits_test = logits_t

    # Compute loss.
    reconstruct_loss = F.cross_entropy(logits.reshape(-1, indexer.size), target.reshape(-1))
    reconstruct_loss_test = F.cross_entropy(logits_test.reshape(-1, indexer.size), target.reshape(-1))

    loss = (1-recon_alpha)* (reconstruct_loss + kl_loss) + recon_alpha * reconstruct_loss_test 

    if training:
        optimizer.zero_grad() #to clear Variable's grad section to be 0
        loss.backward() #to calculate the gradient of every Variable
        optimizer.step() #to update all the Variable's value, using Variable's grad section and the lr value

    return loss.item()


def train(train_source_path, train_target_path,
          valid_source_path, valid_target_path,
          info_dir,
          embed_size, hidden_size, latent_size,
          number_highway_layers, number_rnn_layers,
          batch_size, test_batch_size, number_epochs, 
          learning_rate, dropout_rate, enforce_ratio,
          save_path, load_path, glove_path,
          valid_size=200, valid_every=10000, 
          train_print_every=1000, valid_print_every=100, args=None):
    """Train a Nested Variation Autoencoder (Gupta/18).
    
    Args:
        train_source_path: path to training data source file (each line is a string sentence).
        train_target_path: same as above, but for target file.
        valid_source_path: same as above, but for validation-source file.
        valid_target_path: same as above, but for validation-target file.
        info_dir: folder to read indexer and save configuration.
        embed_size: word embedding size.
        hidden_size: RNN hidden size.
        latent_size: latent/reparametric size.
        number_highway_layers: number of layers of highway net.
        number_rnn_layers: number of layers of stacked RNN.
        batch_size: batch size.
        number_epochs: #passes over data.
        learning_rate: (starting) learning rate.
        dropout_rate: dropout rate.        
        enforce_ratio: ratio of teacher enforce (during training).
        save_path: path to save model.
        laod_path: path to load model.
        glove_path: path to GloVe `.txt` file.
        valid_size: validation size.
        valid_every: validate after every `valid_every` global steps.
        train_print_every: print out loss after every `train_print_every` global steps.
        valid_print_every: print out loss after every `valid_print_every` valid steps.
    """
    indexer = dill.load(open(info_dir + "indexer2.p", "rb"))
    configs = {"embed_size": embed_size,
               "hidden_size": hidden_size,
               "latent_size": latent_size,
               "number_highway_layers": number_highway_layers,
               "number_rnn_layers": number_rnn_layers}
    dill.dump(configs, open(info_dir + "configs.p", "wb"))
    train_iterator = ParaphraseIterator(train_source_path, train_target_path, indexer)
    vocab_size = indexer.size
    start_index = indexer.get_index("<s>", add=False)
    end_index = indexer.get_index("</s>", add=False)
    if glove_path is None:
        glove_init = None
    else:
        glove_init = load_glove(glove_path, indexer, embed_size)
    model = NestedVAE(embed_size, hidden_size, latent_size, vocab_size,
                       number_highway_layers, number_rnn_layers,
                       dropout_rate, enforce_ratio, 
                       start_index, end_index, glove_init, alpha=args.alpha).to(DEVICE)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0
    current_epoch = 0
    best_valid_loss = np.inf
    best_bleu = -100
    best_rouge = -100
    best_meteor = -100
    best_cider = -100
    best_ter = np.inf

    start = time.time()

    if args.test:
        test(valid_iterator, indexer, test_batch_size, valid_size, model)
        return


    ###################################################################
    # pretrain vae.
    ###################################################################
    num_sents = 0
    print('start to pretrain vae')
    while train_iterator.epoch < number_epochs:
        print("\nEpoch %d\n" % (current_epoch + 1))
        model.train()
        train_losses = []
        while train_iterator.epoch <= current_epoch:
            global_step += 1
            batch = train_iterator.get_next_batch(batch_size)
            recon_alpha = args.recon_alpha * min(  global_step / args.wp_recon_alpha , 1.0)
            train_loss = run_batch(model, optimizer, batch, indexer,
                                   global_step, recon_alpha=recon_alpha, training=True)
            train_losses.append(train_loss)
            num_sents += batch[0].size(0)

            if global_step % train_print_every == 0:
                print("Step %d loss = %.4f (%.2f elapsed).  Throughput: %.2f examples/sec \n" % (global_step, 
                                                                 np.mean(train_losses),
                                                                 time.time() - start,
                       						num_sents / (time.time() - start)))
                start = time.time()
                train_losses = []

            if global_step % valid_every == 0:
                print("\n##### Running validation ... #####\n")
                score_rouge_best, average_valid_loss, score_bleu, score_meteor, score_rouge, score_cider, score_ter = test(valid_source_path, valid_target_path, indexer, test_batch_size, valid_size, model)
                train_losses = []
                if average_valid_loss < best_valid_loss:
                    print("Saving model weights for the best valid loss %.6f during vae pretraining" % average_valid_loss)
                    best_valid_loss = average_valid_loss
                    torch.save(model.state_dict(), save_path.replace('nest','loss'))
                    print("Saved as %s\n" % save_path)
                if score_bleu[0] > best_bleu:
                    print("Saving model weights for the best bleu %.6f during vae pretraining" % score_bleu[0])
                    best_bleu = score_bleu[0]
                    torch.save(model.state_dict(), save_path.replace('nest','best_bleu'))
                    print("Saved as %s\n" %save_path+'_best_bleu')
                """
                if score_rouge_best['rouge-1']['f'] > best_rouge:
                    print("Saving model weights for the best rouge-1 F-1 %.6f during vae pretraining" % score_rouge_best['rouge-1']['f'])
                    best_rouge = score_rouge_best['rouge-1']['f']
                    torch.save(model.state_dict(), save_path.replace('nest','best_rouge_1_F_1'))
                    print("Saved as %s\n" %save_path+'_best_rouge_1_F_1')
                """
                if score_meteor > best_meteor:
                    print("Saving model weights for the best meteor %.6f during vae pretraining" % score_meteor)
                    best_meteor = score_meteor
                    torch.save(model.state_dict(), save_path.replace('nest','best_meteor'))
                    print("Saved as %s\n" %save_path+'_best_meteor')
                if score_cider > best_cider:
                    print("Saving model weights for the best cider %.6f during vae pretraining" % score_cider)
                    best_cider = score_cider
                    torch.save(model.state_dict(), save_path.replace('nest','best_cider'))
                    print("Saved as %s\n" %save_path+'_best_cider')
                if score_ter < best_ter:
                    print("Saving model weights for the best ter %.6f during vae pretraining" % score_ter)
                    best_ter = score_ter
                    torch.save(model.state_dict(), save_path.replace('nest','best_ter'))
                    print("Saved as %s\n" %save_path+'_best_ter')
                print('best bleu {} best meteor {} best ter {}'.format(best_bleu, best_meteor, best_ter))
                model.train()                    
        current_epoch += 1   

    ###################################################################
    # adv training 
    ###################################################################
    print('pre training vae  is done !!!!!!!!!!!')
    print('adv training vaegan !!!!!!!!!!!')
    disc = Discriminator(embed_size, hidden_size, number_rnn_layers, dropout_rate) 
    disc.cuda()
    optimizer_critic = torch.optim.Adam(disc.critic.parameters(), lr=args.adv_lr)
    optimizer_disc = torch.optim.Adam([p for (n,p) in disc.named_parameters() if 'critic' not in n], lr=args.adv_lr)
    start = time.time()
    adv_start_step = global_step

    while train_iterator.epoch < args.adv_epochs:
        print("\nADV Epoch %d\n" % (current_epoch - number_epochs + 1))
        disc.train()
        model.train()

        recon_losses, kl_losses, gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, avg_accs= [[] for _ in range(10)]
        num_sents = 0

        while train_iterator.epoch <= current_epoch:
            global_step += 1
            batch = train_iterator.get_next_batch(batch_size)
            source, target = batch
            num_sents += source.size(0)
            recon_alpha = args.recon_alpha * min( (global_step-adv_start_step) / args.wp_recon_alpha , 1.0)
            gamma = args.gamma * min((global_step -adv_start_step)/ args.wp_gamma , 1.0)

            if global_step % args.adv_ratio != 0 and (global_step-adv_start_step) > args.wp_disc:
                # train vae / generator 
                fake_logits_t, kl_loss = model(source, target, True)
                fake_logits, fake_logits_test = fake_logits_t
                # Compute vae loss.
                reconstruct_loss = F.cross_entropy(fake_logits.reshape(-1, indexer.size), target.reshape(-1))
                reconstruct_loss_test = F.cross_entropy(fake_logits_test.reshape(-1, indexer.size), target.reshape(-1))
                recon_losses += [reconstruct_loss.data.item()*(1-recon_alpha) + recon_alpha*reconstruct_loss_test.data.item()]
                kl_losses += [kl_loss.data.item()]
                vae_loss = (reconstruct_loss + kl_loss)*(1-recon_alpha) + recon_alpha*reconstruct_loss_test

                # compute gen loss
                gen_fake_logits, kl_loss = model(source, target, False)
                fake_sentence = torch.argmax(gen_fake_logits, dim=2) 
                fake_out, fake_baseline = disc(fake_sentence, model.embedder, model.highway)
                cumulative_rewards = get_cumulative_rewards(fake_out)
                gen_loss = reinforce_gen_loss(cumulative_rewards, gen_fake_logits, fake_sentence, fake_baseline,
                    use_baseline=args.use_baseline, beta=args.beta_gan, adv_clip=args.adv_clip)
                gen_losses += [gen_loss.data.item()]

                total_loss =  vae_loss + gamma*gen_loss        
                apply_loss(optimizer, total_loss, clip_norm=args.grad_clip, retain_graph = True)

            else:
                # train disc
                # train disc on real data
                real_out, _  = disc(source, model.embedder, model.highway)
                real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                p_real = torch.sigmoid(real_out)
                real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data.item()
                p_real = p_real.mean().data.item()
                ps_real += [p_real]
                real_accs += [real_acc]

                # train disc on fake data
                fake_logits, _ = model(source, target, False)
                fake_sentence = torch.argmax(fake_logits, dim=2) 

                fake_out, fake_baseline  = disc(fake_sentence.detach(), model.embedder, model.highway)
                fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                p_fake = torch.sigmoid(fake_out)
                fake_acc = (p_fake[:, -1] < 0.5).type(torch.float).mean().data.item()
                p_fake = p_fake.mean().data.item()
                ps_fake += [p_fake]
                fake_accs += [fake_acc]

                avg_accs += [(fake_acc+real_acc)/2]
                disc_loss = (fake_loss + real_loss) / 2
                disc_losses += [disc_loss.data.item()]
                apply_loss(optimizer_disc, disc_loss, clip_norm=args.grad_clip)

                if args.use_baseline:
                    cumulative_rewards = get_cumulative_rewards(fake_out)
                    critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline) 
                    critic_losses += [critic_loss.data.item()]
                    apply_loss(optimizer_critic, critic_loss, clip_norm=args.grad_clip)


            if global_step % train_print_every == 0:
                print('Step: %d, Recon_loss: %.2f, KL_loss: %.2f, Gan_Gen_loss: %.2f, Gan_Disc_loss: %.2f, Gan_Critic_loss: %.2f, Real_Acc: %.2f, Fake_Acc: %.2f, Avg_Acc: %.2f, Throughput: %.2f examples/sec' % 
                      (global_step, np.mean(recon_losses), np.mean(kl_losses), np.mean(gen_losses), 
                       np.mean(disc_losses), np.mean(critic_losses),
                       np.mean(real_accs), np.mean(fake_accs), np.mean(avg_accs),
                       num_sents / (time.time() - start)))

                start = time.time()
                recon_losses, kl_losses, gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, avg_accs= [[] for _ in range(10)]
                num_sents = 0

            if global_step % valid_every == 0:
                print("\n##### Running validation ... #####\n")

                score_rouge_best, average_valid_loss, score_bleu, score_meteor, score_rouge, score_cider, score_ter = test(valid_source_path, valid_target_path, indexer, test_batch_size, valid_size, model)

                if average_valid_loss < best_valid_loss:
                    print("Saving model weights for best valid loss %.6f" % average_valid_loss)
                    best_valid_loss = average_valid_loss
                    torch.save(model.state_dict(), save_path+'_adv')
                    print("Saved as %s\n" % save_path+'_adv')
                if score_bleu[0] > best_bleu:
                    print("Saving model weights for the best bleu %.6f during VAEGAN" % score_bleu[0])
                    best_bleu = score_bleu[0]
                    torch.save(model.state_dict(), save_path.replace('nest','best_bleu'+'_adv'))
                    print("Saved as %s\n" %save_path+'_best_bleu'+'_adv')
                """
                if score_rouge_best['rouge-1']['f'] > best_rouge:
                    print("Saving model weights for the best rouge-1 F-1 %.6f during VAEGAN" % score_rouge_best['rouge-1']['f'])
                    best_rouge = score_rouge_best['rouge-1']['f']
                    torch.save(model.state_dict(), save_path.replace('nest','best_rouge_1_F_1'+'_adv'))
                    print("Saved as %s\n" %save_path+'_best_rouge_1_F_1'+'_adv')
                """
                if score_meteor > best_meteor:
                    print("Saving model weights for the best meteor %.6f during VAEGAN" % score_meteor)
                    best_meteor = score_meteor
                    torch.save(model.state_dict(), save_path.replace('nest','best_meteor'+'_adv'))
                    print("Saved as %s\n" %save_path+'_best_meteor'+'_adv')
                """
                if score_cider > best_cider:
                    print("Saving model weights for the best cider %.6f during VAEGAN" % score_cider)
                    best_cider = score_cider
                    torch.save(model.state_dict(), save_path.replace('nest','best_cider'+'_adv'))
                    print("Saved as %s\n" %save_path+'_best_cider'+'_adv')
                """
                if score_ter < best_ter:
                    print("Saving model weights for the best ter %.6f during VAEGAN" % score_ter)
                    best_ter = score_ter
                    torch.save(model.state_dict(), save_path.replace('nest','best_ter'+'_adv'))
                    print("Saved as %s\n" %save_path+'_best_ter'+'_adv')
                print('best bleu {} best meteor {} best ter {}'.format(best_bleu, best_meteor, best_ter))
                model.train()                    
        current_epoch += 1   

        

def test(valid_source_path, valid_target_path, indexer, batch_size, valid_size, model):
    valid_iterator = ParaphraseIterator(valid_source_path, valid_target_path, indexer)
    model.eval()
    valid_losses = []

    metric_bleu = Bleu(4)
    metric_meteor = Meteor()
    #metric_rouge = Rouge()
    #metric_cider = Cider()
    res_dict = {}
    gt_dict = {}
    dict_id = 0

    valid_step = 0
    for valid_step in range(valid_size):
        if valid_iterator.epoch > 0:
            break

        batch = valid_iterator.get_next_batch(batch_size)
        source, target = batch
        
        with torch.no_grad():
            logits, kl_loss = model(source, target, False)
        # Compute loss.
        reconstruct_loss = F.cross_entropy(logits.reshape(-1, indexer.size), target.reshape(-1))
        valid_loss = reconstruct_loss.item()

        # get sentences
        if valid_step == 0:
            res, gt = get_pairs(indexer, logits, target, True)
        else:
            res, gt = get_pairs(indexer, logits, target)

        for s1, s2 in zip(res, gt):
            res_dict[dict_id] = [s1]
            gt_dict[dict_id] = [s2]
            dict_id += 1

        valid_losses.append(valid_loss)
        valid_step += 1


    average_valid_loss = np.mean(valid_losses)

    score_bleu, _ = metric_bleu.compute_score(gt_dict, res_dict)
    score_meteor, _ = metric_meteor.compute_score(gt_dict, res_dict)
    #score_rouge, _ = metric_rouge.compute_score(gt_dict, res_dict)
    #score_cider, _ = metric_cider.compute_score(gt_dict, res_dict)
    score_cider = 0
    score_rouge = 0

    # write to file.
    gt_f = open('tmp_gt.txt', 'w')
    res_f = open('tmp_res.txt', 'w')
    for key in res_dict.keys():
        gt_f.write(gt_dict[key][0]+'\n')
        res_f.write(res_dict[key][0]+'\n')
    
    #ter = 0
    gt_f.close()
    res_f.close()
    ter = compute_ter('tmp_res.txt', 'tmp_gt.txt')



    # 02/13, new rouge score
    """
    gt_f = open('tmp_gt.txt', 'r')
    res_f = open('tmp_res.txt', 'r')
    gt_str = gt_f.readlines()
    res_str = res_f.readlines()
    gt_f.close()
    res_f.close()

    eval_rouge_best = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=4, limit_length=True, length_limit=100, 
            length_limit_type='words', apply_best=True, alpha=0.5, weight_factor=0.2, stemming=True)
    score_rouge_best = eval_rouge_best.get_scores(res_str, gt_str)
    """
    score_rouge_best = 0


    print('VALID loss = %.4f, bleu(4): %.4f, meteor: %.4f, rouge: %.4f, cider: %.4f, ter: %.4f' % (average_valid_loss, 
        score_bleu[3], score_meteor, score_rouge, score_cider, ter))
    print('bleus ', score_bleu)
    print('rouge best', score_rouge_best)

    return score_rouge_best, average_valid_loss, score_bleu, score_meteor, score_rouge, score_cider, ter


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_source_path", type=str, default='MSCOCO/train_source.txt')
    parser.add_argument("--train_target_path", type=str, default='MSCOCO/train_target.txt')
    parser.add_argument("--valid_source_path", type=str, default='MSCOCO/valid_source.txt')
    parser.add_argument("--valid_target_path", type=str, default='MSCOCO/valid_source.txt')
    parser.add_argument("--info_dir", type=str, default='MSCOCO/info/')
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--latent_size", type=int, default=400)
    parser.add_argument("--number_highway_layers", type=int, default=2)
    parser.add_argument("--number_rnn_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--number_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--enforce_ratio", type=float, default=0.8)
    parser.add_argument("--save_path", type=str, default='SavedModels/nest-vae-test.ckpt')
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--glove_path", type=str)
    parser.add_argument("--valid_size", type=int, default=2000)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--train_print_every", type=int, default=100)
    parser.add_argument("--valid_print_every", type=int, default=100)

    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--recon_alpha", type=float, default=0.5, help='encoder in the test time for the reconstruction loss')
    parser.add_argument("--alpha", type=float, default=0.7, help='temperature for softmax in the reconstruction loss')
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument('--adv_epochs', default=100, type=int)
    parser.add_argument('--adv_lr', default=1e-6, type=float)
    parser.add_argument('--adv_ratio', default=5, type=float) # steps ratio disc/gen
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--adv_clip', type=float, default=5.)
    parser.add_argument('--use_baseline', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--beta_gan', type=float, default=0.)

    parser.add_argument("--wp_recon_alpha", type=int, default=20000)
    parser.add_argument("--wp_gamma", type=int, default=20000)
    parser.add_argument("--wp_disc", type=int, default=1000)

    args = parser.parse_args()
    
    print("Device:", DEVICE)
    print("Configs:", args)
    print("\n>>>>>>>>>> START TRAINING <<<<<<<<<<\n")
    train(args.train_source_path,
          args.train_target_path,
          args.valid_source_path,
          args.valid_target_path,
          args.info_dir,
          args.embed_size,
          args.hidden_size,
          args.latent_size,
          args.number_highway_layers,
          args.number_rnn_layers,
          args.batch_size,
          args.test_batch_size,
          args.number_epochs,
          args.learning_rate,
          args.dropout_rate,
          args.enforce_ratio,
          args.save_path,
          args.load_path,
          args.glove_path,
          args.valid_size,
          args.valid_every,
          args.train_print_every,
          args.valid_print_every, 
          args)    
