# Recover filesystem on Jetson Tegra K1

You can recover the whole filesystem on Jetson TK1 if the system suffers from unexpected behaviour or is broken.
This tutorial summarises how to do system recovery on built-in eMMC for Jetson TK1.


## Prepare recovery for TK1 on host

On host Linux computer, you need to prepare latest driver and Linux filesystem before recovery.

First, download tarballs from NVIDIA,

```sh
mkdir -p /tmp/l4t
cd /tmp/l4t

# download driver
wget http://developer.download.nvidia.com/embedded/L4T/r21_Release_v3.0/Tegra124_Linux_R21.3.0_armhf.tbz2

# download Linux filesystem
wget http://developer.download.nvidia.com/embedded/L4T/r21_Release_v3.0/Tegra_Linux_Sample-Root-Filesystem_R21.3.0_armhf.tbz2
```

And extract tarballs,

```sh
cd /tmp/l4t
sudo tar xpf Tegra124_Linux_R21.3.0_armhf.tbz2
cd Linux_for_Tegra/rootfs
sudo tar xpf ../../Tegra_Linux_Sample-Root-Filesystem_R21.3.0_armhf.tbz2
cd ..
sudo ./apply_binaries.sh
```

Then, you will be able to see `Success!` message.


## Recover TK1 drivers and filesystem

Now bring your TK1 next to the host computer.
Jetson TK1 board, micro-USB cable and power supply are needed.

1. Connect TK1 board to the host USB port (directly connected to host's mother board).
2. Press and hold `FORCE RECOVERY` button, located on the corner of TK1.
3. Press the `RESET` button (if the board was already on), located next to the `FORCE RECOVERY` one, or the `POWER` button (if the board was off).
4. Wait a few seconds and release `FORCE RECOVERY` button.

On host, type the command to check if the board got into recovery mode.

```sh
lsusb | grep -i nvidia
```

The command will return the message below if the host recognizes the TK1 board.

```
Bus 002 Device 004: ID 0955:7140 NVidia Corp.
```

Then use the command to inject files to the TK1 board.

```sh
sudo ./flash.sh -S 8GiB jetson-tk1 mmcblk0p1
```

When you'll see the following output on screen means that the recovery is finished.
Your board will boot and you can now disconnect the USB cable.

```
Time taken for flashing xxx Secs
*** The target ardbeg has been flashed successfully. ***
```


## Set up basic configurations on TK1 device

You need to install CUDA toolkit for L4T in order to use CUDA compiler.
Tutorial [here](Install-CUDA-6.5-on-Jetson-TK1.md).

Then, set up basic configurations at your preference on TK1 device.
First, allow community-maintained open-source softwares.

```
sudo add-apt-repository universe
```

Then,

```sh
sudo date --set="Fri Nov 21 01:23:45 EST 2014"
sudo apt-get update
sudo apt-get install -y git tmux vim curl
```

Disable GUI desktop at boot-up

```sh
sudo mv /etc/init/lightdm.conf /etc/init/lightdm.conf.disabled
```
