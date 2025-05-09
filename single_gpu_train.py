import inspect
from generator_simple import GeneratorSimple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
import os
import os.path as osp
import argparse
import numpy as np
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import accuracy_score
from generator_simple import GeneratorSimple
from discriminator_simple import DiscriminatorSimple
from data_loader import DATAReader
from compose_models import get_cnn, get_msresnet, get_rawboost, get_rawnet2, get_resnet, get_sentence_transformer, get_ssdnet, get_inc_ssdnet, get_wav2vec2_model, get_speech_to_text_model
from utils import batch_audio_to_mel, batch_mel_to_audio, get_transciption_loss, transcribe_audio, transcribe_s2t
from visualize import compare_audio_samples
from tqdm import tqdm
import datetime



class Trainer:
    def __init__(
        self,
        G: torch.nn.Module,
        D: torch.nn.Module,
        cl_models: dict,
        # cl_model1: torch.nn.Module,
        # cl_model2: torch.nn.Module,
        # cl_model3: torch.nn.Module,
        t_processor_1, 
        t_model_1: torch.nn.Module,
        # t_processor_2,
        # t_model_2: torch.nn.Module,
        sentence_transformer: torch.nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        # scheduler_G:torch.optim.lr_scheduler,
        # scheduler_D:torch.optim.lr_scheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        save_dir_path: str,
        save_output: str,
        device
    ) -> None:
        self.device = device # torch.device(f"cuda:{gpu_id}")
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.cl_models = {name: model.to(self.device) for name, model in cl_models.items()}
        # self.cl_model1 = cl_model1.to(self.device)
        # self.cl_model2 = cl_model2.to(self.device)
        # self.cl_model3 = cl_model3.to(self.device)
        self.t_processor_1 = t_processor_1        
        self.t_model_1 = t_model_1.to(self.device)
        # self.t_processor_2 = t_processor_2 
        # self.t_model_2 = t_model_2.to(self.device)
        self.sentence_transformer = sentence_transformer.to(self.device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_output = save_output
        # Loss functions
        self.perceptual_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.scaler = torch.GradScaler()
        self.s_w = { 
            # "ssdnet":0.0001,
            # "inc_ssdnet":0.0001,
            "rawboost":0.0001,
            "rawnet2":0.0001,
            # "resnet":0.0001,
            # "msresnet":0.0001,
            # "cnn":0.0001,
            }
        self.t1_w = 0.0001
        s_w_str = "_".join([f"{value:.6f}" for value in self.s_w.values()])
        self.save_dir_path = f"{save_dir_path}_{s_w_str}_{self.t1_w}"



    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)


    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def sLoss(self,x, y, model):
        logits = model(x)
        return self.classification_loss(logits, y.to(dtype=torch.long))

    def train(self, epoch: int):
        self.G.train()
        # self._run_epoch(epoch)
        # if epoch % self.save_every == 0:
        #     self._save_checkpoint(epoch)
        g_losses, d_losses,  t1_losses, t2_losses = [], [], [], []
        c_losses_dict = {name: [] for name in self.s_w}  # Create a dictionary for classification losses


        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)
        for index, train_sample in enumerate(progress_bar):
            real = train_sample[0].unsqueeze(1).to(self.device, dtype=torch.float)
            forged = train_sample[2].unsqueeze(1).to(self.device, dtype=torch.float)
            y_real = torch.zeros(real.shape[0]).to(self.device, dtype=torch.float)
            y_fake = torch.ones(forged.shape[0]).to(self.device, dtype=torch.float)

            # Train Generator, Discriminator
            self.optimizer_G.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
                assert self.G.training, "Error: Generator is in eval mode!"

                attacked = self.G(forged)

                forged_transciption1 = transcribe_audio(forged,self.t_processor_1, self.t_model_1, self.device)
                attacked_transciption1 = transcribe_audio(attacked,self.t_processor_1, self.t_model_1, self.device)

                # forged_transciption2 = transcribe_s2t(forged,self.t_processor_2,self.t_model_2,self.device)
                # attacked_transciption2 = transcribe_s2t(attacked,self.t_processor_2,self.t_model_2,self.device)

                if index == 0 and self.save_output in "yes":
                    forged_audio = forged[0].detach()  # Select first sample of forged audio
                    attacked_audio = attacked[0].detach()   # Corresponding generated audio

                    # self.s1_w, self.s2_w, self.t1_w
                    # Plot and compare
                    wav_dir_path = 'Wav_Plot_' + self.save_dir_path
                    os.makedirs(wav_dir_path, exist_ok=True)
                    compare_audio_samples(forged_audio, attacked_audio, forged_transciption1, attacked_transciption1, epoch, index,wav_dir_path, sr=16000) # forged_transciption2, attacked_transciption2,


                t1_loss = get_transciption_loss(forged_transciption1, attacked_transciption1, self.sentence_transformer)
                # t2_loss = get_transciption_loss(forged_transciption2,attacked_transciption2)

                per_loss = self.perceptual_loss(forged, attacked)
                adv_loss = self.adversarial_loss(y_real, self.D(attacked).squeeze())
                c_losses = {name: self.sLoss(attacked, y_real.to(dtype=torch.long), model) for name, model in self.cl_models.items()}
                # c1_loss = self.sLoss(attacked, y_real.to(dtype=torch.long),self.cl_model1)
                # c2_loss = self.sLoss(attacked, y_real.to(dtype=torch.long),self.cl_model2)
                # c3_loss = self.sLoss(attacked.squeeze(1), y_real.to(dtype=torch.long),self.cl_model3)   # used fake label just to revert the label  for rawnet
                # c3_loss = np.float32(c3_loss.item())   # required for rawnet_2

                # print(f"C losses: {', '.join([f'{name}: {loss:.4f}' for name, loss in c_losses.items()])}: T1 loss: {t1_loss}")#, T2 loss: {t2_loss}")
                g_loss = 0.01*per_loss + adv_loss + sum(self.s_w[name] * c_losses[name].item() for name in self.s_w) + self.t1_w*t1_loss #+ self.s3_w*c3_loss

            # g_loss.backward()
            # self.optimizer_G.step()
            # Scale the loss and backpropagate
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()


            self.optimizer_D.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                real_loss = self.adversarial_loss(y_real, self.D(real).squeeze())
                fake_loss = self.adversarial_loss(self.D(attacked.detach()).squeeze(), y_fake)

                # fake_loss = self.adversarial_loss(y_fake, self.D(attacked).squeeze())
                d_loss = (real_loss + fake_loss) / 2

            # d_loss = d_loss.clone()  # To ensure it's not modified

            # d_loss.backward()
            # self.optimizer_D.step()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_D)
            self.scaler.update()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            # Append losses dynamically
            for name, loss in c_losses.items():
                c_losses_dict[name].append(loss.item())
            t1_losses.append(t1_loss.item())

            # Update tqdm progress bar with current losses
            # progress_bar.set_postfix({
            #     "G_Loss": f"{np.mean(g_losses):.4f}",
            #     **{f"{name}_Loss": f"{np.mean(losses):.4f}" for name, losses in c_losses_dict.items()},

            #     "D_Loss": f"{np.mean(d_losses):.4f}",
            #     "T1_Loss": f"{np.mean(t1_losses):.4f}",
            # })

        progress_bar.close()  # Ensure tqdm closes cleanly when done
        return (np.mean(g_losses),
                *(np.mean(losses) for losses in c_losses_dict.values()),  # Unpack classification losses dynamically
                np.mean(d_losses),
                np.mean(t1_losses))
    
    def cal_acc(self, y, x, model):
        # outputs = inception(x)
        # with torch.no_grad(): 
        outputs = model(x)   # (x.squeeze(1))[1] for aasist   # ssdnet_model, assist_model 
        outputs = nn.Softmax(dim=-1)(outputs)
        _, y_ = torch.max(outputs, 1)

        acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

        return acc

    def test(self,epoch=0):
        self.G.eval()
        # sampler = DistributedSampler(test_dataset)
        accuracies = {name: {"real": [], "fake": [], "af": []} for name in self.cl_models.keys()}

        progress_bar = tqdm(self.test_loader, desc=f"[Test] Epoch {epoch}", unit="batch", leave=True)

        for index,test_sample in enumerate(progress_bar):
            real = test_sample[0].unsqueeze(1).to(self.device, dtype=torch.float)
            forged = test_sample[2].unsqueeze(1).to(self.device, dtype=torch.float)

            y_real = torch.zeros(real.shape[0]).to(self.device, dtype=torch.float)
            y_fake = torch.ones(forged.shape[0]).to(self.device, dtype=torch.float)

            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
                fake = self.G(forged)

                forged_transciption1 = transcribe_audio(forged,self.t_processor_1, self.t_model_1, self.device)
                attacked_transciption1 = transcribe_audio(fake,self.t_processor_1, self.t_model_1, self.device)

                if index <= 1  and self.save_output in "yes":
                    forged_audio = forged[0].detach()  # Select first sample of forged audio
                    attacked_audio = fake[0].detach()   # Corresponding generated audio

                    # self.s1_w, self.s2_w, self.t1_w
                    # Plot and compare
                    wav_dir_path = 'Wav_Plot_Tes_' + self.save_dir_path
                    os.makedirs(wav_dir_path, exist_ok=True)
                    compare_audio_samples(forged_audio, attacked_audio, forged_transciption1, attacked_transciption1, epoch, index,wav_dir_path, sr=16000)


                # for model_name, model in zip(
                #     ["cl_model1", "cl_model2", "cl_model3"], [self.cl_model1, self.cl_model2, self.cl_model3]
                # ):
                for model_name, model in self.cl_models.items():                    
                    with torch.no_grad():  # No gradients needed during testing
                        # if model_name in 'cl_model3':
                        #     accuracies[model_name]["real"].append(self.cal_acc(y_real, real.squeeze(1), model))          # used fake label just to revert the label  for rawnet
                        #     accuracies[model_name]["fake"].append(self.cal_acc(y_fake, forged.squeeze(1), model))   
                        #     accuracies[model_name]["af"].append(self.cal_acc(y_fake, fake.squeeze(1), model))
                        # else:
                        model.eval()
                        accuracies[model_name]["real"].append(self.cal_acc(y_real, real, model))
                        accuracies[model_name]["fake"].append(self.cal_acc(y_fake, forged, model))
                        accuracies[model_name]["af"].append(self.cal_acc(y_fake, fake, model))

        progress_bar.close()

        # Compute mean accuracies
        results = {
            model_name: {
                metric: 100 * np.mean(values) for metric, values in acc_dict.items()
            }
            for model_name, acc_dict in accuracies.items()
        }

        return results



def load_optims(G,D,lr):
    # Optimizers and schedulers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.SGD(D.parameters(), lr=lr)
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.9)
    return optimizer_G,optimizer_D,scheduler_G,scheduler_D


def load_models(device):
    # train_set = MyTrainDataset(2048)  # load your dataset
    # model = torch.nn.Linear(20, 1)  # load your model
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Define models
    G = GeneratorSimple()
    D = DiscriminatorSimple()
    # G = DDP(G, device_ids=[local_rank])
    # D = DDP(D, device_ids=[local_rank])
    # Load classification models
    cl_models = {
        # "ssdnet": get_ssdnet('original', device),
        # "inc_ssdnet": get_inc_ssdnet('original', device),
        "rawboost": get_rawboost(device),
        "rawnet2": get_rawnet2(device),
        # "resnet": get_resnet(device),
        # "msresnet": get_msresnet(device),
        # "cnn": get_cnn(device)
    }
    # cl_model1 = get_ssdnet('original', device) #DDP(get_ssdnet(device), device_ids=[local_rank])
    # cl_model2 = get_inc_ssdnet('original', device) #DDP(get_inc_ssdnet(device), device_ids=[local_rank])
    # cl_model3 = get_resnet(device) #DDP(get_inc_ssdnet(device), device_ids=[local_rank])
    
    t_processor_1, t_model_1 = get_wav2vec2_model(device)
    # t_processor_2, t_model_2 = get_speech_to_text_model(device)
    sentence_transformer = get_sentence_transformer(device)

    return G,D,cl_models,t_processor_1, t_model_1,sentence_transformer  # t_processor_2,t_model_2,



def prepare_dataloader(batch_size, n_workers):
    train_dataset = DATAReader(split='TRAIN')
    # sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, #sampler=sampler,
        num_workers=n_workers, pin_memory=True, drop_last=True
    )

    test_dataset = DATAReader(split='TEST')
    # sampler = DistributedSampler(test_dataset)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,#sampler=sampler, 
                num_workers=n_workers, pin_memory=True, drop_last=True)
    return train_loader,test_loader


def main(nEpochs, batch_size,lr,num_workers, save_output, device_id):
    import datetime
    model_name = "ssd_incS_"
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir_path = f"{time_now}_{model_name}_with_T"
    # device = torch.device(f"cuda:{device_id}")
    device = torch.device('cuda', device_id)

    print(f"GPU Device initialized: {str(device)}")


    G,D,cl_models,t_processor_1, t_model_1,sentence_transformer = load_models(device)
    optimizer_G,optimizer_D,scheduler_G,scheduler_D = load_optims(G,D,lr)
    train_loader,test_loader = prepare_dataloader(batch_size, num_workers)

    trainer = Trainer(G,D,cl_models,t_processor_1, t_model_1,sentence_transformer,\
                      optimizer_G,optimizer_D,train_loader,test_loader,save_dir_path,save_output, device)
    checkpoint_dir_path = 'CHECKPOINTS_'+trainer.save_dir_path

    for epoch in range(nEpochs):

        # Write the entire file's code during the first epoch
        if epoch == 0 and save_output in "yes":
            # Get the path of the current script
            os.makedirs(checkpoint_dir_path, exist_ok=True)
            file_path = os.path.abspath(__file__)

            # Read the entire file and write it to the results file
            with open(file_path, 'r') as f:
                main_file_content = f.read()

            # Read the entire file and write it to the results file
            with open("./generator_simple.py", 'r') as f:
                generator_file_content = f.read()

            new_file_path = osp.join(checkpoint_dir_path, 'epoch_10_code.txt')

            with open(new_file_path, 'w') as f:
                f.write("### Entire File Code ###\n")
                f.write(main_file_content)
                f.write("\n\n")
                f.write("\n\n### GeneratorSimple Class Code ###\n")
                f.write(generator_file_content)  # Write the generator class code
                f.write("\n\n")
            # del generator_file_content

        losses = trainer.train(epoch)  # Get all returned losses dynamically

        g_loss = losses[0]  # Generator loss
        *c_losses, d_loss, t1_loss = losses[1:]  # Unpack classification losses dynamically

        scheduler_G.step()
        scheduler_D.step()

        c_losses_str = " ".join([f"[C{i+1} loss: {c_loss:.6f}]" for i, c_loss in enumerate(c_losses)])

        print(f"[Train] [Epoch {epoch}/{args.nEpochs}], [LR: G={scheduler_G.get_last_lr()[0]:f}, D={scheduler_D.get_last_lr()[0]:f}], "
            f"{c_losses_str} [D loss: {d_loss:.6f}], [G loss: {g_loss:.6f}], [T1 loss: {t1_loss:.6f}]")

        results = trainer.test(epoch)

        # Iterate through classifiers dynamically
        for model_name, acc_dict in results.items():
            r_acc, f_acc, af_acc = acc_dict.values()  # Extract accuracy values
            print(f'[Test {model_name}] [Epoch {epoch}/{args.nEpochs}], [Acc: {r_acc:.2f}, {f_acc:.2f}, {af_acc:.2f}]')

        # Save output if enabled
        if save_output in "yes":
            checkpoints = {
                'epoch': epoch + 1,
                'state_dict': G.state_dict()
            }

            # Ensure checkpoint directory exists
            os.makedirs(checkpoint_dir_path, exist_ok=True)

            # Save the generator model
            torch.save(checkpoints, osp.join(checkpoint_dir_path, f'generator_{epoch+1}.pth'))

            # Save test accuracies to a file
            results_file = osp.join(checkpoint_dir_path, 'test_accuracies.txt')
            with open(results_file, 'a') as f:
                for model_name, acc_dict in results.items():
                    r_acc, f_acc, af_acc = acc_dict.values()
                    f.write(f"{model_name}: Epoch {epoch + 1}/{args.nEpochs}: Acc=({r_acc:.2f}, {f_acc:.2f}, {af_acc:.2f})\t\t")
                f.write("\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpochs', '-epoch', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    args.save_output = "yes"
    device_id = 3
    main(args.nEpochs, args.batch_size,args.lr,args.num_workers, args.save_output, device_id)

# with 
