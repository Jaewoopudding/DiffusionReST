import torch
from ddpo_pytorch.diffusers_patch.ddim_with_kl import predict_x0_from_xt_search, ddim_step_KL_search
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob


class Node:
    def __init__(self, state, reward, timestep, log_prob=0, ref_log_prob=0, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.reward = reward
        self.timestep = timestep
        self.value = 0
        self.best_reward = None
        self.log_prob = log_prob
        self.ref_log_prob = ref_log_prob

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def add_children(self, states, timesteps, log_probs=None, ref_log_probs=None):
        if log_probs is None:
            log_probs = [None] * len(states)
        if ref_log_probs is None:
            ref_log_probs = [None] * len(states)
        for state, timestep, lp, rlp in zip(states, timesteps, log_probs, ref_log_probs):
            self.children.append(Node(state=state, reward=None, timestep=timestep, log_prob=lp, ref_log_prob=rlp, parent=self))

    def _terminal_checker(self, max_timestep):
        return self.timestep == max_timestep
        

class BatchedNode:
    def __init__(self, node_list):
        self.node_list = node_list

    @property
    def batch_size(self):
        return len(self.node_list)

    @property
    def states(self):
        return torch.stack([node.state for node in self.node_list], dim=0)

    @states.setter
    def states(self, new_states):
        for i, node in enumerate(self.node_list):
            node.state = new_states[i : i + 1].squeeze()

    @property
    def timesteps(self):
        return torch.stack([node.timestep for node in self.node_list], dim=0)

    @timesteps.setter
    def timesteps(self, new_timesteps):
        for i, node in enumerate(self.node_list):
            node.timestep = new_timesteps[i : i + 1]

    @property
    def rewards(self):
        return torch.tensor([node.reward for node in self.node_list])

    @rewards.setter
    def rewards(self, new_rewards):
        for i, node in enumerate(self.node_list):
            node.reward = new_rewards[i : i + 1]

    @property
    def log_probs(self):
        return torch.stack([node.log_prob for node in self.node_list], dim=0)

    @log_probs.setter
    def log_probs(self, new_log_probs):
        for i, node in enumerate(self.node_list):
            node.log_prob = new_log_probs[i : i + 1]

    @property
    def ref_log_probs(self):
        return torch.stack([node.ref_log_prob for node in self.node_list], dim=0)

    @ref_log_probs.setter
    def ref_log_probs(self, new_ref_log_probs):
        for i, node in enumerate(self.node_list):
            node.ref_log_prob = new_ref_log_probs[i : i + 1]

    def get_children(self):
        return [node.get_children() for node in self.node_list]

    def get_novel_children(self):
        return [
            [child for child in node.get_children() if child.reward is None]
            for node in self.node_list
        ]

    def add_children(self, children_states_list, children_timesteps_list, children_log_probs_list=None, children_ref_log_probs_list=None):
        if children_log_probs_list is None:
            children_log_probs_list = [None] * len(self.node_list)
        if children_ref_log_probs_list is None:
            children_ref_log_probs_list = [None] * len(self.node_list)
        for node, states, ts, lp, rlp in zip(self.node_list, children_states_list, children_timesteps_list, children_log_probs_list, children_ref_log_probs_list):
            node.add_children(states, ts, lp, rlp)

    def __call__(self):
        return self.node_list

    def __getitem__(self, idx):
        return self.node_list[idx]

class SearchPolicy:
    def __init__(
            self,
            initial_children,
            pipeline,
            do_classifier_free_guidance,
            reward_fn,
            config,
            prompt_embeds=None,
            cross_attention_kwargs=None,
            guidance_scale=1.0,
            eta=1.0,
            prompt=None,
            prompt_metadata=None,
            ref_unet=None,
            gamma=0.93,
        ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.config = config

        self.base_unet = pipeline.unet if config.search.hill_climbing else ref_unet
        self.ref_unet = ref_unet
        self.reward_fn = reward_fn

        self.prompt_embeds = prompt_embeds
        self.cross_attention_kwargs = cross_attention_kwargs
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.prompt = prompt
        self.prompt_metadata = prompt_metadata

        self.gamma = gamma
        self.search_kl = torch.tensor(
            config.search.search_kl, device=self.device, dtype=torch.float32
        )
        self.max_timestep = pipeline.scheduler.timesteps[-1]

        self.root_nodes = BatchedNode([Node(state=None, timestep=None, parent=None, reward=None)])
        n_children = config.search.duplicate if config.initial_search else 1
        self.root_nodes.add_children(
            initial_children.view(1, n_children, *initial_children.shape[1:]),
            torch.ones(1, n_children, device=self.device) * pipeline.scheduler.timesteps[0],
        )
        for nodes in zip(*self.root_nodes.get_novel_children()):
            self.evaluate(BatchedNode(nodes))
            self.backpropagate(nodes)

    # ── helpers ──────────────────────────────────────────────

    def _predict_noise(self, unet, latent, timesteps):
        """UNet forward + classifier-free guidance → guided noise prediction."""
        if self.do_classifier_free_guidance:
            model_input = torch.cat([latent] * 2, dim=0)
        else:
            model_input = latent
        model_input = self.pipeline.scheduler.scale_model_input(model_input, timesteps).to(unet.dtype)

        noise_pred = unet(
            model_input,
            timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else timesteps,
            encoder_hidden_states=self.prompt_embeds,
            cross_attention_kwargs=self.cross_attention_kwargs,
        ).sample

        if self.do_classifier_free_guidance:
            uncond, text = noise_pred.chunk(2, dim=0)
            noise_pred = uncond + self.guidance_scale * (text - uncond)
        return noise_pred

    def _decode_to_image(self, pred_original_sample):
        """VAE decode → postprocess to image tensor."""
        image = self.pipeline.vae.decode(
            pred_original_sample.to(self.pipeline.vae.dtype) / self.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return self.pipeline.image_processor.postprocess(
            image, output_type="pt", do_denormalize=[True] * image.shape[0]
        )

    def _gather_child_stats(self, node):
        """Collect child-node statistics as tensors for a selection function."""
        children = node.get_children()
        d = self.device
        stats = dict(
            parent_visits=torch.tensor(node.visit_count, dtype=torch.float32, device=d),
            values=torch.tensor([c.value for c in children], dtype=torch.float32, device=d).squeeze(),
            visits=torch.tensor([c.visit_count for c in children], dtype=torch.float32, device=d).squeeze(),
            rewards=torch.tensor([c.reward for c in children], dtype=torch.float32, device=d).squeeze(),
            log_probs=torch.tensor(
                [c.log_prob if c.log_prob is not None else 0 for c in children], dtype=torch.float32, device=d
            ).squeeze(),
            ref_log_probs=torch.tensor(
                [c.ref_log_prob if c.log_prob is not None else 0 for c in children], dtype=torch.float32, device=d
            ).squeeze(),
            timestep=torch.as_tensor(
                node.timestep if node.timestep is not None else self.pipeline.scheduler.config.num_train_timesteps
            ).to(d, dtype=torch.float32),
        )
        return children, stats

    def _select_child(self, node, select_fn):
        """Apply a selection function and return (chosen_child, index)."""
        children, s = self._gather_child_stats(node)
        idx = select_fn(
            s["parent_visits"], s["values"], s["visits"],
            s["rewards"], s["log_probs"], s["ref_log_probs"], s["timestep"],
        )
        return children[idx], idx

    # ── core search operations ────────────────────────────────

    def select(self, select_fn):
        """Descend from root via select_fn until a leaf or terminal node."""
        selected = []
        for node in self.root_nodes:
            current = node
            while current.get_children() and not current._terminal_checker(self.max_timestep):
                current, _ = self._select_child(current, select_fn)
            selected.append(current)
        return BatchedNode(selected)

    def expand(self, nodes, use_gradient=False):
        """Generate children via DDIM step, evaluate them, and backpropagate."""
        scheduler = self.pipeline.scheduler
        step_offset = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        duplicate = self.config.search.duplicate

        grad_mode = torch.enable_grad() if use_gradient else torch.no_grad()
        with grad_mode:
            latent = nodes.states.detach().to(self.pipeline.unet.dtype)
            if use_gradient:
                latent.requires_grad_(True)
            timesteps = nodes.timesteps.to(self.pipeline.unet.dtype)

            noise_pred = self._predict_noise(self.base_unet, latent, timesteps)

            new_latents, pred_original_sample, variance_coeff, _, _, log_probs = ddim_step_KL_search(
                scheduler, noise_pred, noise_pred, timesteps, latent,
                eta=self.eta, duplicate=duplicate,
            )
            new_latents = new_latents.view(self.pipeline.batch_size, duplicate, *new_latents.shape[1:])

            if use_gradient:
                image = self._decode_to_image(pred_original_sample)
                evaluation, _ = self.reward_fn(image, self.prompt, self.prompt_metadata)
                guidance = torch.autograd.grad(
                    outputs=evaluation.to(torch.float32), inputs=latent,
                    grad_outputs=torch.ones_like(evaluation),
                )[0].detach() / self.config.search.search_kl
                if torch.isnan(guidance).any():
                    guidance = torch.nan_to_num(guidance, nan=0)
                latent = latent.detach()

                discount = self.gamma ** (
                    scheduler.num_inference_steps
                    - (scheduler.config.num_train_timesteps - nodes.timesteps) // step_offset
                )
                new_latents = new_latents + variance_coeff * guidance * discount.view(-1, 1, 1, 1)

            new_timesteps = torch.clamp(timesteps - step_offset, min=0)
            new_timesteps = new_timesteps.repeat_interleave(duplicate).view(1, duplicate)

            if self.ref_unet is not None:
                ref_noise_pred = self._predict_noise(self.base_unet, latent, timesteps).detach()
                _, ref_log_probs = ddim_step_with_logprob(
                    self=scheduler,
                    model_output=ref_noise_pred,
                    timestep=timesteps.to(torch.int64),
                    sample=latent.repeat_interleave(duplicate, dim=0),
                    eta=self.eta,
                    prev_sample=new_latents.squeeze(0),
                )
                nodes.add_children(
                    new_latents.detach(), new_timesteps,
                    log_probs.view(1, duplicate).detach(),
                    ref_log_probs.view(1, duplicate).detach(),
                )
                del ref_noise_pred, ref_log_probs
            else:
                nodes.add_children(
                    new_latents.detach(), new_timesteps,
                    log_probs.view(1, duplicate).detach(),
                )

            for child_nodes in zip(*nodes.get_novel_children()):
                self.evaluate(BatchedNode(child_nodes))
                self.backpropagate(child_nodes)

        del latent, noise_pred
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, batched_nodes):
        """Decode latents → image → reward, then store reward in nodes."""
        states = batched_nodes.states
        timesteps = batched_nodes.timesteps.to(self.pipeline.unet.dtype)

        noise_pred = self._predict_noise(self.base_unet, states, timesteps)
        pred_original_sample = predict_x0_from_xt_search(
            self.pipeline.scheduler, noise_pred, timesteps, states
        )
        image = self._decode_to_image(pred_original_sample)
        evaluation, _ = self.reward_fn(image, self.prompt, self.prompt_metadata)
        batched_nodes.rewards = evaluation
        return evaluation

    def backpropagate(self, children):
        """Propagate reward from children up to root."""
        for child in children:
            r = child.reward
            current = child
            while current is not None:
                current.visit_count += 1
                current.value += r
                if current.best_reward is None or r > current.best_reward:
                    current.best_reward = r
                current = current.get_parent()

    def _free_subtree(self, node):
        for child in node.get_children():
            self._free_subtree(child)
        if hasattr(node, "state") and isinstance(node.state, torch.Tensor):
            if node.state.is_cuda:
                node.state.detach()
            node.state = None
        node.children.clear()
        node.parent = None

    def act_and_prune(self, select_fn, prune=True):
        """Select best child from root, advance, optionally free unselected subtrees."""
        selected = []
        for node in self.root_nodes:
            current = node
            children = current.get_children()
            if children:
                best_child, best_idx = self._select_child(current, select_fn)
                if prune:
                    for i, child in enumerate(children):
                        if i != best_idx:
                            self._free_subtree(child)
                current = best_child
            selected.append(current)
        self.root_nodes = BatchedNode(selected)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── selection functions (passed as select_fn) ────────────

    @torch.no_grad()
    def importance_sampling(self, parent_visits, values, visits, rewards, log_probs, ref_log_probs, timestep):
        """Softmax sampling: w ∝ exp(r/α · γ^t) · π_ref/π."""
        scheduler = self.pipeline.scheduler
        step_offset = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        discount = self.gamma ** (
            scheduler.num_inference_steps
            - (scheduler.config.num_train_timesteps - timestep) // step_offset
        )
        log_w = torch.nan_to_num(rewards) / self.search_kl * discount + ref_log_probs - log_probs
        log_w = log_w - log_w.max(dim=0, keepdim=True)[0]
        return torch.distributions.Categorical(logits=log_w.view(-1)).sample()

    def argmax_value(self, parent_visits, values, visits, rewards=None, log_probs=None, ref_log_probs=None, timestep=None):
        """Select child with highest accumulated value."""
        return torch.argmax(values).item()