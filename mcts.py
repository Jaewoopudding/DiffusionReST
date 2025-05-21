import torch, torchvision
import numpy as np
from tqdm import tqdm
from ddpo_pytorch.diffusers_patch.ddim_with_kl import predict_x0_from_xt_MCTS, ddim_step_KL_MCTS
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
tqdm.tqdm = lambda *args, **kwargs: args[0]

class Node:
    def __init__(self, state, reward, timestep, log_prob=0, ref_log_prob=0, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0 ## TODO fix
        self.prior = 0
        self.reward = reward
        self.timestep = timestep 
        self.value = 0
        self.best_reward = None
        self.log_prob = log_prob
        self.ref_log_prob = ref_log_prob
        self.gradient = None
        
    def get_parent(self):
        return self.parent
    
    def get_children(self):
        return self.children
        
    def add_children(self, states, timesteps, log_probs=None, ref_log_probs=None):
        if log_probs is None and ref_log_probs is None:
            for state, timestep in zip(states, timesteps):
                self.children.append(Node(state=state, reward=None, timestep=timestep, log_prob=None, ref_log_prob=None, parent=self))
        elif log_probs is not None and ref_log_probs is None:
            for state, timestep, log_prob in zip(states, timesteps, log_probs):
                self.children.append(Node(state=state, reward=None, timestep=timestep, log_prob=log_prob, ref_log_prob=None, parent=self))
        elif log_probs is None and ref_log_probs is not None:
            for state, timestep, ref_log_prob in zip(states, timesteps, ref_log_probs):
                self.children.append(Node(state=state, reward=None, timestep=timestep, log_prob=None, ref_log_prob=ref_log_prob, parent=self))
        else:
            for state, timestep, log_prob, ref_log_prob in zip(states, timesteps, log_probs, ref_log_probs):
                self.children.append(Node(state=state, reward=None, timestep=timestep, log_prob=log_prob, ref_log_prob=ref_log_prob, parent=self))
            
    def set_value(self, value):
        self.value = value
        
    def _terminal_checker(self, max_timestep):
        if self.timestep == max_timestep:
            return True
        else:
            return False
        

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
        assert new_states.shape[0] == self.batch_size, "Batch size mismatch in states setter."
        for i, node in enumerate(self.node_list):
            node.state = new_states[i : i + 1].squeeze()

    @property
    def timesteps(self):
        # 각 노드의 timestep을 (B, ...) 형태로 결합 (노드마다 shape이 다를 수 있으므로 cat dim은 상황에 맞게 수정)
        return torch.stack([node.timestep for node in self.node_list], dim=0)

    @timesteps.setter
    def timesteps(self, new_timesteps):
        # new_timesteps: (B, ...) 텐서라고 가정
        assert new_timesteps.shape[0] == self.batch_size, "Batch size mismatch in timesteps setter."
        for i, node in enumerate(self.node_list):
            node.timestep = new_timesteps[i : i + 1]
            
    @property
    def rewards(self):
        # 각 노드의 reward를 (B, ...) 형태로 결합합니다.
        return torch.tensor([node.reward for node in self.node_list])

    @rewards.setter
    def rewards(self, new_rewards):
        # new_rewards: (B, ...) 텐서라고 가정합니다.
        assert new_rewards.shape[0] == self.batch_size, "Batch size mismatch in rewards setter."
        for i, node in enumerate(self.node_list):
            node.reward = new_rewards[i : i + 1]
            
    @property
    def best_rewards(self):
        # 각 노드의 reward를 (B, ...) 형태로 결합합니다.
        return torch.tensor([node.best_reward for node in self.node_list])
            
    @property
    def values(self):
        return torch.tensor([node.value for node in self.node_list])
    
    @property
    def visit_counts(self):
        return torch.tensor([node.visit_count for node in self.node_list])
    

    @property
    def states(self):
        return torch.stack([node.state for node in self.node_list], dim=0)
    
    @property
    def log_probs(self):
        return torch.stack([node.log_prob for node in self.node_list], dim=0)
    
    @log_probs.setter   
    def log_probs(self, new_log_probs):
        assert new_log_probs.shape[0] == self.batch_size, "Batch size mismatch in log_probs setter."
        for i, node in enumerate(self.node_list):
            node.log_prob = new_log_probs[i : i + 1]    
            
    def get_children(self):
        return [node.get_children() for node in self.node_list]
    
    def get_novel_children(self):
        result = []
        for node in self.node_list:
            novel_children = []
            children = node.get_children()
            for child in children:
                if child.reward is None:
                    novel_children.append(child)
            result.append(novel_children)
        return result
            
    def add_children(self, children_states_list, children_timesteps_list, children_log_probs_list=None, children_ref_log_probs_list=None):
        if children_log_probs_list is None and children_ref_log_probs_list is None:
            [node.add_children(states, ts) for node, states, ts in zip(self.node_list, children_states_list, children_timesteps_list)]
        elif children_log_probs_list is not None and children_ref_log_probs_list is None:
            [node.add_children(states, ts, log_probs) for node, states, ts, log_probs in zip(self.node_list, children_states_list, children_timesteps_list, children_log_probs_list)]
        elif children_log_probs_list is None and children_ref_log_probs_list is not None:
            [node.add_children(states, ts, ref_log_probs=ref_log_probs) for node, states, ts, ref_log_probs in zip(self.node_list, children_states_list, children_timesteps_list, children_ref_log_probs_list)]
        else:   
            [node.add_children(states, ts, log_probs, ref_log_probs) for node, states, ts, log_probs, ref_log_probs in zip(self.node_list, children_states_list, children_timesteps_list, children_log_probs_list, children_ref_log_probs_list)]
        
    def __call__(self):
        return self.node_list 
    
    def __getitem__(self, idx):
        return self.node_list[idx]

class TreePolicy:
    def __init__(
            self, 
            initial_children, 
            select_function, 
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
            gamma=0.93
        ):
        self.prompt_embeds = prompt_embeds
        self.cross_attention_kwargs = cross_attention_kwargs
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.select_function = select_function
        self.progressive_widening = config.search.progressive_widening
        self.pw_alpha = config.search.pw_alpha
        self.pipeline = pipeline
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.expansion_coef = config.search.expansion_coef 
        self.exploration_constant = config.search.expansion_coef  # UCT 상수로 사용
        self.max_timestep = self.pipeline.scheduler.timesteps[-1] 
        self.kl_lagrangian_coef = torch.tensor(config.search.kl_lagrangian_coef, device=pipeline.device).to(torch.float32)
        self.tempering_gamma = config.search.tempering_gamma
        self.lookforward_fn = lambda r: r / self.kl_lagrangian_coef
        self.reward_fn = reward_fn
        # initial_children: torch.Tensor of shape (B * duplicate, C, H, W)
        node_list = [Node(state=None, timestep=None, parent=None, reward=None)]
        self.device = pipeline.device
        self.pipeline_config = config
        
        self.prompt = prompt
        self.prompt_metadata = prompt_metadata
        
        self.ref_unet = ref_unet
        self.gamma = gamma
        
        self.base_unet = pipeline.unet if config.search.hill_climbing else ref_unet
        
        # initial node for x_T starting point
        self.root_nodes = BatchedNode(node_list)
        self.initial_nodes = self.root_nodes
        self.root_nodes.add_children(
            initial_children.view(1, config.search.duplicate * config.search.nfe_per_action, *initial_children.shape[1:]), 
            torch.ones(1, config.search.duplicate * config.search.nfe_per_action, device=self.device) * pipeline.scheduler.timesteps[0]
        )
        
        for nodes in tqdm(list(zip(*self.root_nodes.get_novel_children())), desc='Initial Evaluating', leave=False, position=2, disable=True):
            self.evaluate(BatchedNode(nodes))
            self.backpropagate(nodes)


    def select(self, select_fn):
        """
        각 트리의 root부터 시작하여, 각 트리마다 leaf(또는 terminal) 노드까지 UCT 선택을 진행합니다.
        어떤 트리는 다른 트리보다 일찍 선택 과정이 끝날 수 있으므로, 각 트리에 대해 개별적으로 처리한 후
        결과를 BatchedNode 객체로 반환합니다.
        """
        selected_nodes = []
        # self.root_nodes.node_list는 배치 내 개별 root Node들의 리스트입니다.
        for node in self.root_nodes:
            current = node
            # 자식이 존재하고 현재 노드가 leaf(terminal이 아님) 상태이면 계속 진행
            while current.get_children() and not current._terminal_checker(self.max_timestep):
                if self.progressive_widening and (current.visit_count ** self.pw_alpha >= len(current.get_children())) and current.parent is not None:
                    break
                
                parent_visits_tensor = torch.tensor(current.visit_count, dtype=torch.float32, device=self.device)
                child_values = torch.tensor([child.value for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_visits = torch.tensor([child.visit_count for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_rewards = torch.tensor([child.reward for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_log_likelihood = torch.tensor([child.log_prob if child.log_prob is not None else 0 for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_ref_log_likelihood = torch.tensor([child.ref_log_prob if child.log_prob is not None else 0  for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                current_timesteps = torch.as_tensor(current.timestep if current.timestep is not None else self.pipeline.scheduler.config.num_train_timesteps).to(self.device, dtype=torch.float32)            
                    
                best_idx =  select_fn(parent_visits_tensor, child_values, child_visits, child_rewards, child_log_likelihood, child_ref_log_likelihood, current_timesteps)
                best_child = current.get_children()[best_idx]
                
                current = best_child
            selected_nodes.append(current)
        # 선택된 노드들을 BatchedNode로 묶어 반환합니다.
        return BatchedNode(selected_nodes)
            
    
    def expand(self, nodes, use_gradient=False, jump=None):
        """
        확장 단계: BatchedNode인 nodes의 각 노드(state)를 바탕으로 자식 노드들을 생성합니다.
        여기서는 각 노드가 pipeline.duplicate 만큼의 자식을 갖는다고 가정합니다.
        t와 latents는 현재 timestep과 latent 정보를 의미합니다.
        """
        # nodes.states: (B, C, H, W)
        step_offset = self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
        duplicate = self.pipeline_config.search.duplicate
        jump_step = None
        jump_latents = [None] * duplicate
        jump_timesteps = None
        
        grad_mode = torch.enable_grad() if use_gradient else torch.no_grad()
        with grad_mode:
            latent = nodes.states.detach().to(self.pipeline.unet.dtype)
            if use_gradient:
                latent.requires_grad_(True)
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latent] * 2, dim=0)  # (2B, C, H, W)
            else:
                latent_model_input = latent  # (B, C, H, W)
            
            # scale_model_input는 배치 지원
            timesteps = nodes.timesteps.to(self.pipeline.unet.dtype)
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, timesteps).to(self.pipeline.unet.dtype)

            
            noise_pred = self.base_unet(
                latent_model_input,
                timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else timesteps,
                encoder_hidden_states=self.prompt_embeds,
                cross_attention_kwargs=self.cross_attention_kwargs,
            ).sample

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                old_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            noise_pred = old_noise_pred 
                
            # ddim_step_KL_modified: 노드의 상태에서 새로운 latent 후보들을 생성
            # new_latents: (B * duplicate, C, H, W)
            if jump:
                jump_step = timesteps / 2
                jump_timesteps = torch.clamp(timesteps - jump_step, min=0)

            model_output = noise_pred
            new_latents, jump_latents, pred_original_sample, variance_coeff, jump_variance_coeff, _, _, log_probs = ddim_step_KL_MCTS( ## TODO 입력 noise_pred 확인
                self.pipeline.scheduler,
                model_output,    # 예측된 노이즈
                old_noise_pred,
                timesteps,
                latent,
                eta=self.eta,
                duplicate=duplicate,
                jump_step=jump_step,
            ) # (B * duplicate, C, H, W)  
            
            new_latents = new_latents.view(self.pipeline.batch_size, duplicate, *new_latents.shape[1:])
            
            if use_gradient:
                image = self.pipeline.vae.decode(pred_original_sample.to(self.pipeline.vae.dtype) / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
                do_denormalize = [True] * image.shape[0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)
                
                evaluation, _ = self.reward_fn(image, self.prompt, self.prompt_metadata)
                evaluation = self.lookforward_fn(evaluation).to(torch.float32)
                
                guidance = torch.autograd.grad(outputs=evaluation, inputs=latent, grad_outputs=torch.ones_like(evaluation))[0].detach()
                if torch.isnan(guidance).any():
                    guidance = torch.nan_to_num(guidance, nan=0)
                    evaluation = torch.nan_to_num(evaluation, nan=-1e6)
                latent = latent.detach()
                
                discount = self.gamma ** (self.pipeline.scheduler.num_inference_steps - (self.pipeline.scheduler.config.num_train_timesteps - nodes.timesteps) // step_offset - 1)
                # min_scale = torch.tensor([min((1 + self.tempering_gamma) ** (((self.pipeline.scheduler.timesteps[0] - timesteps) // step_offset) + 1) - 1, 1.)] * timesteps.shape[0], device=self.device)
                # min_scale_next = torch.tensor([min((1 + self.tempering_gamma) ** (((self.pipeline.scheduler.timesteps[0] - timesteps) // step_offset) + 2) - 1, 1.)] * timesteps.shape[0], device=self.device)
                
                
                new_latents = new_latents + variance_coeff * guidance * discount.view(-1, 1, 1, 1)
                if jump:
                    jump_latents = jump_latents + jump_variance_coeff * guidance * discount.view(-1, 1, 1, 1)
                model_output = noise_pred + variance_coeff * guidance * discount.view(-1, 1, 1, 1)

            if jump:
                jump_latents = [None] * duplicate
            new_timesteps = timesteps - step_offset
            mask = new_timesteps >= 0
            new_timesteps = torch.where(
                mask,
                new_timesteps, 
                torch.zeros_like(new_timesteps, device=new_timesteps.device) 
            ).repeat_interleave(duplicate).view(1, duplicate)

            if self.ref_unet is not None:
                ref_noise_pred = self.base_unet(
                    latent_model_input,
                    timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else timesteps,
                    encoder_hidden_states=self.prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                ).sample
                ref_noise_pred = ref_noise_pred.detach()
                model_output = ref_noise_pred
            
                if self.do_classifier_free_guidance:
                    ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2, dim=0)
                    ref_old_noise_pred = ref_noise_pred_uncond + self.guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)
                    model_output = ref_old_noise_pred

                _, ref_log_probs = ddim_step_with_logprob(
                    self=self.pipeline.scheduler,
                    model_output=model_output,
                    timestep=timesteps.to(torch.int64),
                    sample=latent.repeat_interleave(duplicate, dim=0),
                    eta=self.eta,
                    prev_sample=new_latents.squeeze(0)
                )
                ref_log_probs = ref_log_probs.detach()

                nodes.add_children(new_latents.detach(), new_timesteps, log_probs.view(1, duplicate).detach(), ref_log_probs.view(1, duplicate).detach())
                del ref_noise_pred, ref_log_probs
            else:
                nodes.add_children(new_latents.detach(), new_timesteps, log_probs.view(1, duplicate).detach())

            for idx, nodes in enumerate(tqdm(list(zip(*nodes.get_novel_children())), desc='Evaluating', leave=False, position=2, disable=True)):
                self.evaluate(BatchedNode(nodes))
                self.backpropagate(nodes)

        del latent, old_noise_pred, model_output, noise_pred
        torch.cuda.empty_cache()
    
    
    @torch.no_grad()
    def evaluate(self, batched_nodes, jump_latents=None, jump_timesteps=None):
        if (jump_latents == None) and (jump_timesteps is None):
            states = batched_nodes.states
            timesteps = batched_nodes.timesteps
        elif (jump_latents is not None) and (jump_timesteps is not None):
            states = jump_latents.unsqueeze(0)
            timesteps = jump_timesteps
        else:
            raise ValueError("Both jump_latents and jump_timesteps should be None or both should be not None.")
        latent_model_input = torch.cat([states] * 2) 
        latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, timesteps)
        
        latent_model_input = latent_model_input.to(self.pipeline.unet.dtype)
        timesteps = timesteps.to(self.pipeline.unet.dtype)
        
        noise_pred = self.base_unet(
            latent_model_input, 
            timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else timesteps, 
            encoder_hidden_states= self.prompt_embeds, 
            cross_attention_kwargs=self.cross_attention_kwargs
        ).sample    
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        new_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) 
        pred_original_sample = predict_x0_from_xt_MCTS(
                            self.pipeline.scheduler,
                            new_noise_pred,   
                            timesteps,
                            states
        )
        
        # if timesteps == 1:
        #     breakpoint()
        image = self.pipeline.vae.decode(pred_original_sample.to(self.pipeline.vae.dtype) / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)
        evaluation, _ = self.reward_fn(image, self.prompt, self.prompt_metadata)
        batched_nodes.rewards = evaluation
        return evaluation
        
    def backpropagate(self,  children):
        for child in children:
            r = child.reward  
            current = child
            while current is not None:
                current.visit_count += 1
                current.value += r
                if (current.best_reward) is None or (r > current.best_reward):
                    current.best_reward = r
                current = current.get_parent()

    def _free_subtree(self, node):
        for child in node.get_children():
            self._free_subtree(child)
        if hasattr(node, 'state') and isinstance(node.state, torch.Tensor):
            if node.state.is_cuda:
                node.state.detach()
            node.state = None
        node.children.clear()
        node.parent = None

    def act_and_prune(self, select_fn, prune=True):
        selected_nodes = []
        for node in self.root_nodes:
            current = node
            children = current.get_children()
            if children: 
                parent_visits_tensor = torch.tensor(current.visit_count, dtype=torch.float32, device=self.device)
                child_values = torch.tensor([child.value for child in children], dtype=torch.float32, device=self.device).squeeze()
                child_visits = torch.tensor([child.visit_count for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_rewards = torch.tensor([current.reward for current in children], dtype=torch.float32, device=self.device).squeeze()
                child_log_likelihood = torch.tensor([child.log_prob if child.log_prob is not None else 0 for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                child_ref_log_likelihood = torch.tensor([child.ref_log_prob if child.log_prob is not None else 0  for child in current.get_children()], dtype=torch.float32, device=self.device).squeeze()
                current_timesteps = torch.as_tensor(current.timestep if current.timestep is not None else self.pipeline.scheduler.config.num_train_timesteps).to(self.device, dtype=torch.float32)            
                
                best_idx =  select_fn(parent_visits_tensor, child_values, child_visits, child_rewards, child_log_likelihood, child_ref_log_likelihood, current_timesteps)
                best_child = children[best_idx]
                
                # 선택되지 않은 자식들은 재귀적으로 메모리 해제
                if prune:
                    for i, child in enumerate(children):
                        if i != best_idx:
                            self._free_subtree(child)
                current = best_child
            selected_nodes.append(current)
        self.root_nodes = BatchedNode(selected_nodes)
        # GPU 메모리 비우기
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_final_latent(self):
        return self.root_nodes.states
    
    @torch.no_grad()
    def UCT(self, parent_visits_tensor, child_values, child_visits, child_rewards = None, child_log_likelihood = None, child_ref_log_likelihood = None, current_timesteps = None):
        uct_values = child_values / child_visits + self.exploration_constant * torch.sqrt(torch.log(parent_visits_tensor) / child_visits)
        return torch.argmax(uct_values).item()
    
    @torch.no_grad()
    def importance_sampling(self, parent_visits_tensor, child_values, child_visits, child_rewards, child_log_likelihood, child_ref_log_likelihood, current_timesteps):
        step_offset = self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
        discount = self.gamma ** (self.pipeline.scheduler.num_inference_steps - (self.pipeline.scheduler.config.num_train_timesteps - current_timesteps) // step_offset - 1)

        log_w = child_rewards / self.kl_lagrangian_coef * discount + child_ref_log_likelihood - child_log_likelihood
        log_w = log_w - torch.max(log_w, dim=0, keepdims=True)[0]
        return torch.distributions.Categorical(logits=log_w).sample()

    @torch.no_grad()
    def max_value(self, parent_visits_tensor, child_values, child_visits, child_rewards = None, child_log_likelihood = None, child_ref_log_likelihood = None, current_timesteps = None):
        return torch.argmax(child_values / child_visits).item()
    
    def reset_root_nodes(self):
        self.root_nodes = self.initial_nodes
        
    def argmax_value(self, parent_visits_tensor, child_values, child_visits, child_rewards = None, child_log_likelihood = None, child_ref_log_likelihood = None, current_timesteps = None):
        return torch.argmax(child_values).item()