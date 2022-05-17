import argparse
from agent.sac import SAC
import wandb
import time

wandb.login()
wandb.init(project="crystall generation", entity="kly20")

def run_training(args):
    # config wandb
    wandb.config = {
    "learning_rate_p": args.learning_rate_p,
    "learning_rate_q": args.learning_rate_q,
    "learning_rate_t": args.learning_rate_t,
    "steps": args.num_steps,
    "batch_size": args.batch_size,
    "num_atoms": args.num_atoms,
    "dft_steps": args.dft_steps,}

    # create agent
    if args.agent_type == "standard_sac":
        agent = SAC(args)
        agent.replay_buffer_init(args.buffer_init_size)
    
    # train
    wandb.watch(agent.p_net,log="all",log_freq=5)
    wandb.watch(agent.q_net,log="all",log_freq=5)
    for i in range(args.num_steps):
        tic=time.time()
        q_loss,p_loss,predicted_best_reward,log_probs_mean,q_values_mean,alpha=agent.train_one_step()
        toc=time.time()
        best_reward=agent.expand_buffer(args.expand_buffer_size)
        print("=======================================================================")
        print("epoch ",i," : finished,    time cost : ",toc-tic,"s")
        print("q_loss : ",q_loss,", p_loss : ",p_loss)
        print("max performance : ",best_reward)
        print("=======================================================================")

        # get log
        mu,log_sigma=agent.p_net.forward()
        sigma=log_sigma.exp()

        wandb.log({"q_loss": q_loss,
                    "p_loss": p_loss,
                    "average_log_probs": log_probs_mean.item(),
                    "average_q_values": q_values_mean.item(),
                    "policy_mean": mu.mean().item(),
                    "policy_sigma": sigma.mean().item(),
                    "performance": best_reward,
                    "predicted_performance": predicted_best_reward,
                    "temperature":alpha})
    # save model
    # agent.save_models()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # chemical parameters
    parser.add_argument("--num_atoms",type=int,default=10)
    parser.add_argument("--atom_type_num",type=int,default=8)
    parser.add_argument("--edge_type_num",type=int,default=8)
    parser.add_argument("--feature_dims",type=int,default=4)
    # model parameters
    parser.add_argument("--cuda",action='store_true',default=True)
    parser.add_argument("--num_steps",type=int,default=200)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--expand_buffer_size",type=int,default=64)
    parser.add_argument("--buffer_size",type=int,default=10000)
    parser.add_argument("--buffer_init_size",type=int,default=512)
    parser.add_argument("--dft_steps",type=int,default=200)
    parser.add_argument("--layer_num_p",type=int,default=16)
    parser.add_argument("--layer_num_q",type=int,default=16)
    parser.add_argument("--latent_size",type=int,default=256)
    parser.add_argument("--mlp_hidden_layer_size",type=int,default=256)
    parser.add_argument("--message_pass_num",type=int,default=8)
    parser.add_argument("--dropout_rate",type=float,default=0.2)
    parser.add_argument("--learning_rate_p",type=float,default=1e-5)
    parser.add_argument("--learning_rate_q",type=float,default=1e-4)
    parser.add_argument("--learning_rate_t",type=float,default=1e-6)
    parser.add_argument("--temperature",type=float,default=0.01)
    parser.add_argument("--automatic_entropy_tuning",action='store_true',default=False)
    parser.add_argument("--log_sig_min",type=float,default=-20)
    parser.add_argument("--log_sig_max",type=float,default=2)
    parser.add_argument('--agent_type', choices=['standard_sac', 'new'], default='standard_sac')
    parser.add_argument("--explore_steps",type=int,default=1000)

    args=parser.parse_args()

    run_training(args)
