def myfunc(c, a=1, b=2):
    print(a, b, c)

mydict = {'a': 100, 'b': 200}
myfunc(500, **mydict)

import json
env_args = dict(has_renderer=True, has_offscreen_renderer=True, ignore_done=True, use_camera_obs=True,  
    camera_height=84, camera_width=84, camera_name='agentview', use_object_obs=False, reward_shaping=True)

args = json.dumps(env_args)
print(args)
env_args = json.loads(args)
print(env_args)