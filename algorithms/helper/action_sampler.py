import torch

from dataclass.primitives import BatchedActionOutput



class ActionSampler():
    def __init__(self, args):
        self.args = args
        self.num_samples = 0
        self._init_sampler()

    def _init_sampler(self):
        if self.args.name == "greedy":
            self.sampler = self._greedy_sampler
        elif self.args.name == "epsilon_greedy":
            self.sampler = self._epsilon_greedy_sampler
            self.epsilon_scheduler = self._epsilon_scheduler
        elif self.args.name == "boltzman":
            self.sampler = self._boltzman_sampler
    
    def sample(self, batched_action_output: BatchedActionOutput, accumulate_sample_nums = True) -> BatchedActionOutput:
        if accumulate_sample_nums:
            self.num_samples += batched_action_output.info['action_values'].shape[0]
        return self.sampler(batched_action_output)

    def get_epsilon(self):
        return self._epsilon_scheduler()


    # =====================================
    # Defining Specific Samplers Below
    # =====================================
    def _greedy_sampler(self,  batched_action_output: BatchedActionOutput) -> BatchedActionOutput:
        logits = batched_action_output.info['action_values']
        action = logits.argmax(dim=1, keepdim=True)  # (B, 1)
        
        batched_action_output.action = action
        return batched_action_output
    
    def _epsilon_greedy_sampler(self, batched_action_output: BatchedActionOutput) -> BatchedActionOutput:
        logits = batched_action_output.info['action_values']
        eps = self._epsilon_scheduler()
        batch_size, num_actions = logits.shape[0], logits.shape[1]
        
        greedy = logits.argmax(dim=1, keepdim=True)
        random_actions = torch.randint(0, num_actions, (batch_size, 1), device=logits.device)
        mask = (torch.rand(batch_size, device=logits.device) < eps).unsqueeze(1)
        action = torch.where(mask, random_actions, greedy)
        
        batched_action_output.action = action
        return batched_action_output

    def _boltzman_sampler(self, batched_action_output: BatchedActionOutput) -> BatchedActionOutput:
        logits = batched_action_output.info['action_values']         # B x A
        temperature = float(self.args.extra["temperature"])
        probs = torch.softmax(logits / temperature, dim=1)
        action = torch.multinomial(probs, 1)

        batched_action_output.action = action
        return batched_action_output


    # =====================================
    # Other Helpers
    # =====================================
    def _epsilon_scheduler(self):
        """
        Use self.args.starting_epsilon, self.args.ending_epsilon, and self.args.warmup_steps, and self.args.decay_until_step to compute on the fly (return as float) our epilson for this current step (based on self.num_samples) 
        """
        start = float(self.args.extra["starting_epsilon"])
        end = float(self.args.extra["ending_epsilon"])
        warmup = int(self.args.extra["warmup_steps"])
        total = int(self.args.extra["decay_until_step"])

        if self.num_samples <= warmup:
            return start

        denom = max(1, total - warmup)
        progress = min(1.0, (self.num_samples - warmup) / denom)
        return start + (end - start) * progress