from collections import defaultdict

import cv2
import hydra
import matplotlib.pyplot as plt
from igibson.render.mesh_renderer.mesh_renderer_settings import \
    MeshRendererSettings
from omegaconf import OmegaConf

import ig_navigation
from ig_navigation.igibson_env import SearchEnv


@hydra.main(config_path=ig_navigation.CONFIG_PATH, config_name="config")
def main(cfg):

    cfg.image_width = 512
    cfg.image_height = 512

    env = SearchEnv(
        config_file=OmegaConf.to_object(cfg),
        mode="gui_non_interactive",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
        rendering_settings=MeshRendererSettings(
            enable_pbr=True, enable_shadow=True, msaa=False, hide_robot=cfg.hide_robot
        ),
    )

    for episode in range(10):
        print("Episode: {}".format(episode))
        episode_reward = 0
        state = env.reset()
        reward_breakdown = defaultdict(lambda: 0)

        for idx in range(500):
            # These keys step the simulation
            key = cv2.waitKey()
            if key == 119:
                action = 0
            elif key == 115:
                action = 1
            elif key == 97:
                action = 3
            elif key == 100:
                action = 2
            elif key == 114:
                break
            elif key == 120:
                action = 4
            elif key == 116:
                action = 5
            elif key == 103:
                action = 6
            elif key == 99:
                action = 7
            elif key == 32:
                action = 8
            else:
                action = 4

            state, reward, done, info = env.step(action)

            episode_reward += reward

            # These keys render on the current frame without stepping the simulation
            if key == 112:
                print(f"Episode reward: {episode_reward}")
                print(f"Timestep: {idx}")
                print(f"Timestep reward: {reward}")
                print(f"Episode info: {info}")
                print(f"Episode info: {info}")

            if key == 118:
                _, axs = plt.subplots(1, 2, tight_layout=True, dpi=170)

                if "rgb" in state:
                    axs[0].imshow(state["rgb"])
                    axs[0].set_title("rgb")
                    axs[0].set_axis_off()

                if "depth" in state:
                    axs[1].imshow(state["depth"])
                    axs[1].set_title("depth")
                    axs[1].set_axis_off()

                plt.show()

            if key == 113:
                env.close()
                quit()

            for reward, value in info["reward_breakdown"].items():
                reward_breakdown[reward] += value

            if done:
                break

        print("episode reward: ", episode_reward)
        for key, value in reward_breakdown.items():
            print(f"{key}: {value}")

    env.close()


if __name__ == "__main__":
    main()
