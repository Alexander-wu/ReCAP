#!/usr/bin/env python
"""Enable safe fine-tuning from the released compressive tokenizer.

Upstream train_ctx_tokenizer.py exposes --pretrained_model_name_or_path but the
corresponding branch is a bare `raise NotImplementedError`. It also alternates
generator/discriminator batches before `disc_start`, so half the requested
warm-up steps perform no update. Checkpoint saving is made collective so both
generator-only and adversarial training schedules are safe under Accelerate.

This idempotent patch applies only to the vendored cluster copy.
"""

import argparse
from pathlib import Path


def patch(path, old, new, label):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"{label}: already patched")
        return
    candidates = old if isinstance(old, tuple) else (old,)
    for candidate in candidates:
        if candidate in text:
            path.write_text(text.replace(candidate, new), encoding="utf-8")
            print(f"{label}: patched")
            return
    raise RuntimeError(f"{label}: anchor not found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    path = Path(args.repo) / "train_ctx_tokenizer.py"

    patch(
        path,
        """    elif args.pretrained_model_name_or_path is not None:
        raise NotImplementedError
    else:
        raise NotImplementedError""",
        """    elif args.pretrained_model_name_or_path is not None:
        if args.model_type != "ctx_vqganv64":
            raise NotImplementedError(
                "pretrained loading is implemented only for ctx_vqganv64")
        model = CompressiveVQModelFSQ.from_pretrained(
            args.pretrained_model_name_or_path)
        logger.info(
            "Fine-tuning pretrained tokenizer from " +
            args.pretrained_model_name_or_path)
    else:
        raise NotImplementedError""",
        "pretrained tokenizer loading")

    patch(
        path,
        """            # TODO: make all step generator-step before disc_start
            generator_step = ((i // args.gradient_accumulation_steps) % 2) == 0""",
        """            # Before disc_start every batch must update the tokenizer.
            # Upstream alternated with no-op discriminator batches, so half of
            # max_train_steps performed no learning while still incrementing
            # global_step.
            generator_step = (
                global_step < args.disc_start or
                ((i // args.gradient_accumulation_steps) % 2) == 0)""",
        "all-generator pre-discriminator schedule")

    patch(
        path,
        (
            """                    optimizer.step()
                    lr_scheduler.step()
                    # log gradient norm before zeroing it""",
            """                    optimizer.step()
                    lr_scheduler.step()
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            gen_loss=float(avg_gen_loss),
                            recon=float(avg_recon_loss),
                            perceptual=float(avg_perceptual_loss),
                            ref_recon=float(avg_ref_recon_loss))
                    # log gradient norm before zeroing it""",
        ),
        """                    optimizer.step()
                    lr_scheduler.step()
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            gen_loss=float(avg_gen_loss),
                            recon=float(avg_recon_loss),
                            perceptual=float(avg_perceptual_loss),
                            ref_recon=float(avg_ref_recon_loss))
                        if (global_step + 1) % args.log_steps == 0:
                            accelerator.log({
                                "step_gen_loss": float(avg_gen_loss),
                                "gen_loss/step_recon_loss": float(avg_recon_loss),
                                "gen_loss/step_ref_recon_loss": float(avg_ref_recon_loss),
                                "gen_loss/step_perceptual_loss": float(avg_perceptual_loss),
                                "gen_loss/step_ref_perceptual_loss": float(avg_ref_perceptual_loss),
                                "lr": lr_scheduler.get_last_lr()[0],
                            }, step=global_step + 1)
                    # log gradient norm before zeroing it""",
        "generator loss progress")

    patch(
        path,
        (
            """            # With discriminator disabled every step is a generator
            # step, so the upstream checkpoint block inside the discriminator
            # branch is unreachable. Save here on all processes; Accelerate's
            # save_state is collective in distributed runs.
            if (accelerator.sync_gradients and global_step > 0 and
                    global_step % args.checkpointing_steps == 0):
                save_checkpoint(
                    model, discriminator, args, accelerator,
                    global_step, 'tokenizer')

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:""",
            """            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:""",
        ),
        """            # Save on all processes outside the generator/discriminator
            # branches. Accelerate's save_state is collective in distributed
            # runs, so guarding this call with is_main_process can deadlock.
            if (accelerator.sync_gradients and global_step > 0 and
                    global_step % args.checkpointing_steps == 0):
                save_checkpoint(
                    model, discriminator, args, accelerator,
                    global_step, 'tokenizer')

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:""",
        "generator-only intermediate checkpoints")

    patch(
        path,
        """                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(model, discriminator, args, accelerator, global_step, 'tokenizer')
""",
        """                # Checkpoints are saved collectively below, outside
                # the generator/discriminator branches.
""",
        "collective-only checkpoint saving")

    print("Tokenizer fine-tuning support ready")


if __name__ == "__main__":
    main()
