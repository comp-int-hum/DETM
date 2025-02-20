import copy
import logging
import random
import torch
import numpy
from torch import autograd
import wandb
from .evaluations import original_detm_evaluation


logger = logging.getLogger("utils")

    
def train_model(
        subdocs,
        times,
        model,
        optimizer,
        max_epochs,
        clip=10.0,
        lr_factor=2.0,
        batch_size=32,
        device="cpu",
        val_proportion=0.2,
        detect_anomalies=False,
        use_wandb=False,
        evaluate_while_training=False,
        evaluation_epochs=5,
):
    #times = [model.represent_time(t) for t in times]
    model = model.to(device)
    
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    val_subdocs = [x for x, _ in pairs[:int(val_proportion*len(subdocs))]]

    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    val_times = [x for _, x in pairs[:int(val_proportion*len(times))]]

    logger.info("Saving initial model parameters")
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_val_ppl = float("inf")
    since_annealing = 0
    since_improvement = 0
    
    
    for epoch in range(1, max_epochs + 1):
        logger.info("Starting epoch %d", epoch)
        model.train(True)
        logger.info("Preparing for data")
        model.prepare_for_data(train_subdocs, train_times)
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        word_count = 0

        logger.info("Computing batches")
        indices = torch.randperm(len(train_subdocs))
        indices = torch.split(indices, batch_size)

        
        logger.info("Processing training and updating model")
        for idx, ind in enumerate(indices):

            optimizer.zero_grad()
            model.zero_grad()
            actual_batch_size = len(ind)
            data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
            times_batch = numpy.zeros((actual_batch_size, ))

            for i, doc_id in enumerate(ind):
                subdoc = train_subdocs[doc_id]
                times_batch[i] = train_times[doc_id] #0 if idx > 0 else train_times[doc_id]
                for k, v in subdoc.items():
                    data_batch[i, k] = v
                    word_count += v
                    
            # TODO find solution, preferra
            """if "cETM" in str(type(model)):
                #sort by time
                sorted_indices = numpy.argsort(times_batch)
                data_batch = data_batch[sorted_indices]
                times_batch = times_batch[sorted_indices]"""
            

            data_batch = torch.from_numpy(data_batch).float()
            times_batch = torch.from_numpy(times_batch)
            sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = data_batch / sums
            with autograd.set_detect_anomaly(detect_anomalies):

                loss, nll, kl_alpha, kl_eta, kl_theta = model(
                    data_batch,
                    times_batch,
                )
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += data_batch.shape[0]

            if idx % 100 == 0:
                if use_wandb:
                    wandb.log({
                        "step": (idx) + (epoch-1) * len(indices),
                        "epoch": (epoch-1) + idx / len(indices),
                        "train/loss": torch.sum(loss).item() / data_batch.shape[0],
                        "train/nll": torch.sum(nll).item() / data_batch.shape[0],
                        "train/kl_theta": torch.sum(kl_theta).item() / data_batch.shape[0],
                        "train/kl_eta": torch.sum(kl_eta).item() / data_batch.shape[0],
                        "train/kl_alpha": torch.sum(kl_alpha).item() / data_batch.shape[0]
                    })

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
        lr = optimizer.param_groups[0]['lr']


        logger.info("Processing validation")
        _, val_ppl = apply_model(
            model,
            val_subdocs,
            val_times,
            batch_size,
            detect_anomalies=detect_anomalies
        )
        if evaluate_while_training and epoch % evaluation_epochs == 0:
            logger.info("Evaluating model...")
            topic_diversities, topic_coherences, topic_qualities = original_detm_evaluation(
                model,
                val_subdocs, 
                # TODO check with tom, in the original code the training set was used
                # also the topic coherence is evaluated over the whole dataset, not the respective time slice
            )
            logger.info(
                f"{epoch}: Topic diversities: {topic_diversities}, Topic coherences: {topic_coherences}, Topic qualities: {topic_qualities}"
            )
            if use_wandb:
                wandb_dictionary = {}
                for i, (diversity, coherence, quality) in enumerate(zip(topic_diversities, topic_coherences, topic_qualities)):
                    wandb_dictionary[f"window_{i}/diversity"] = diversity
                    wandb_dictionary[f"window_{i}/coherence"] = coherence
                    wandb_dictionary[f"window_{i}/quality"] = quality
                #wandb_dictionary["epoch"] = epoch
                wandb.log(
                    wandb_dictionary
                )
            logger.info("Evaluation complete.")
        logger.info(
            '{}: LR: {}, Train loss per word: mix_prior={:.2f}, mix={:.2f}, embs={:.2f}, recon={:.2f}, NELBO={:.2f} Val ppl per word: {:.2f}'.format(
                epoch,
                lr,
                cur_kl_eta,
                cur_kl_theta,
                cur_kl_alpha,
                cur_nll,
                cur_loss,
                val_ppl
            )
        )

        if use_wandb:
            wandb.log({
                "val/ppl": val_ppl
            })

        if val_ppl < best_val_ppl:
            logger.info("Copying new best model...")
            best_val_ppl = val_ppl
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            since_improvement = 0
            logger.info("Copied.")
        else:
            since_improvement += 1
        since_annealing += 1
        if since_improvement > 5 and since_annealing > 5 and since_improvement < 10:
            optimizer.param_groups[0]['lr'] /= lr_factor
            model.load_state_dict(best_state)
            since_annealing = 0
        elif numpy.isnan(val_ppl):
            logger.error("Perplexity was NaN: reducing learning rate and trying again...")
            optimizer.load_state_dict(best_optimizer_state)
            model.load_state_dict(best_state)
            optimizer.param_groups[0]['lr'] /= lr_factor
        elif since_improvement >= 10:
            break

    return best_state



def apply_model(
        model,
        subdocs,
        times,
        batch_size=32,
        device="cpu",
        detect_anomalies=False,
        use_wandb=False
):
    model.train(False)
    logger.info("Preparing for data")
    model.prepare_for_data(subdocs, times)

    ppl = 0
    cnt = 0
    indices = torch.randperm(len(subdocs))
    indices = torch.split(indices, batch_size)
    word_count = 0
    
    for idx, ind in enumerate(indices):
        actual_batch_size = len(ind)
        data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
        times_batch = numpy.zeros((actual_batch_size, ))

        for i, subdoc_id in enumerate(ind):
            subdoc = subdocs[subdoc_id]
            tm = times[subdoc_id]
            times_batch[i] = tm
            for k, v in subdoc.items():
                data_batch[i, k] = v
                word_count += v
        data_batch = torch.from_numpy(data_batch).float()
        times_batch = torch.from_numpy(times_batch)
        sums = data_batch.sum(1).unsqueeze(1)
        with autograd.set_detect_anomaly(detect_anomalies):
            loss, nll, kl_alpha, kl_eta, kl_theta = model(
                data_batch,
                times_batch,
            )

            ppl += torch.sum(nll).item()
            cnt += data_batch.shape[0]

    return (), ppl / word_count

