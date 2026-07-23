import time
import mujoco
import mujoco_viewer
import numpy as np


class TrajPlayer:
    def __init__(self, model_path, data_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.001
        # self.model.opt.gravity[2] = -9.81
        self.model.opt.gravity[2] = 0
        self.mj_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.mj_data, 0)

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.mj_data)
        self.viewer.cam.lookat[:] = [0, 0, 1.0]
        self.data_dict = np.load(data_path)
        print(f'Data is load from {data_path}')
        # print(list(self.data_dict.keys()))
        self.num_envs = len(list(self.data_dict.keys())) // 5

        # check if key is in data
        required_keys = ['base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel']
        for env_id in range(self.num_envs):
            for required_key in required_keys:
                required_key = f'{env_id}_{required_key}'
                assert required_key in list(self.data_dict.keys()), f'{required_key} not in data'


    def play(self):
        for env_id in range(self.num_envs):
            print(f'Playing env {env_id} trajectory')
            base_lin_vel_key = f'{env_id}_base_lin_vel'
            base_ang_vel_key = f'{env_id}_base_ang_vel'
            joint_pos_key = f'{env_id}_joint_pos'
            joint_vel_key = f'{env_id}_joint_vel'
            # print(list(self.data_dict.keys()))
            # print(self.data_dict['0_joint_pos'])
            # print(len(self.data_dict['0_joint_pos'][0]))
            # exit(0)
            data_length = len(self.data_dict[joint_pos_key])
            try:
                for i in range(data_length):
                    if self.viewer.is_alive == False:  
                        break
                    base_lin_vel = self.data_dict[base_lin_vel_key][i]
                    base_ang_vel = self.data_dict[base_ang_vel_key][i]
                    joint_pos = self.data_dict[joint_pos_key][i]
                    joint_vel = self.data_dict[joint_vel_key][i]

                    mj_pos, mj_vel = self.isaaclab_to_mujoco(joint_pos, joint_vel)

                    self.mj_data.qpos[7:33] = mj_pos
                    self.mj_data.qvel[0:3] = base_lin_vel
                    self.mj_data.qvel[3:6] = base_ang_vel
                    self.mj_data.qvel[6:32] = mj_vel

                    mujoco.mj_forward(self.model, self.mj_data)
                    self.viewer.render()
                    time.sleep(0.5)

            except KeyboardInterrupt:  
                print("Play interrupted by user")  
      
            finally:  
                self.viewer.close()  
                print("Play finished")


    def isaaclab_to_mujoco(self, joint_pos, joint_vel):
        mapping = [
            0, 4, 8, 12, 16, 20,    # l_leg 1-6
            1, 5, 9, 13, 17, 21,    # r_leg 1-6
            2, 6, 10, 14, 18, 22, 24,  # l_hand 1-7
            3, 7, 11, 15, 19, 23, 25   # r_hand 1-7
        ]

        joint_pos = np.asarray(joint_pos)
        joint_vel = np.asarray(joint_vel)

        mj_pos = joint_pos[mapping]
        mj_vel = joint_vel[mapping]

        mj_pos[0] = 0
        mj_pos[6] = 0

        return mj_pos, mj_vel



if __name__ == '__main__':
    model_path = '/home/sakura/kuavo/kuavo-walk/kuavo-robot-deploy/src/kuavo_assets/models/biped_s45/xml/scene_rl.xml'
    data_path = './test/data.npz'

    player = TrajPlayer(model_path, data_path)
    player.play()


