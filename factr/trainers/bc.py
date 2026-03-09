# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from factr.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        (imgs, obs), actions, mask, labels = batch
        imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask, labels = [ar.to(self.device_id) for ar in (obs, actions, mask, labels)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))

        loss_dict = self.model(imgs, obs, ac_flat, mask_flat, class_labels=labels)

        def reduce_loss(t):
            return t.mean() if t.ndim > 0 else t

        total_loss = reduce_loss(loss_dict["total_loss"])

        if self.is_train:
            self.log("total_loss", global_step, total_loss.item())
            self.log("lr", global_step, self.lr)

            for k, v in loss_dict.items():
                if k in ["logits", "total_loss", "logits"]:
                    continue

                if v is not None:
                    val = reduce_loss(v)
                    if hasattr(val, "item"):
                        val = val.item()

                    self.log(k, global_step, val)

            if loss_dict.get("l1_loss") is not None:
                posterior_l1 = reduce_loss(loss_dict["l1_loss"])
                if hasattr(posterior_l1, "item"):
                    posterior_l1 = posterior_l1.item()
                self.log("posterior_l1", global_step, posterior_l1)

        self.last_train_loss = total_loss.item()

        return total_loss
