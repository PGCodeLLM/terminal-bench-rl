1. Agent and Env have reset called.
Agent just resets state, including messages, current step #, and trajectory
Env resets any state, and also returns an observation (which is at first to be the user prompt), and and metadata, (named info).

2. Agent receives the obsversation and info in update_from_env, and at the core, needs to update its conversation history & also the trajectory with a new step. If first time this is happening, the system message should be included in the self managed conversation history.

3. the agent's chat_completions func is called, which should return the messages for inference, the LLM is then contacted.

4. Agent's update from model is exectuted, here the current step (which was set in update_from_env in step 2 above) has the model_response set, and appends the message to history.

5. Environment receives the step() argument, which receives the content of the LLM's message. It will then exec any actions required and return them as observations, which jumps back to step 2 and loops until the env gets an action string in the step function which means the task is terminated. At that point, the env should return done as true, and provide a reward.