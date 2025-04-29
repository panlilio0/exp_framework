# DevOps: screen manage, SSH、disc manage

This manual tells you how to use pasta lab machine remotely. It is helpful for long duration experiment execution or managing other processes.

## Table of Contents
- [Screen: Terminal Session Management Tool](#Screen)
- [SSH connection](#SSH)
- [Disc management and space check](#DiscManagement)
- [Example of experiment execution](#Example)

## Screen

### What is screen?

Screen is a virtual terminal manager, a UNIX tool that allows you to manage multiple terminal sessions within a single physical terminal. It is especially useful for managing processes that run for long periods of time. If a connection is lost, the process can continue to run and reconnect later.


### Benefits

- Ensures reliable execution over long periods of time, even in environments with unreliable connections
- Efficiently manage multiple sessions from a single terminal
- Log out and continue processing to check your work later
- No network connection loss will affect running processes.

### Basic Command

1. **Start new screen session**
   ```
   screen -S [session name]
   ```
   You can designate session name so that it is easier to check which session is doing what operation.

2. **Display of current screen session list**
   ```
   screen -ls
   ```
   All running Screen sessions and their status are displayed.

3. **Detach from screen session**
   ```
   Ctrl+a, d
   ```
   This allows disconnection while running in the background without terminating the Screen session

4. **Reconnect to screen session**
   ```
   screen -r [session name or id]
   ```
   Specify a session name or ID to reconnect to a specific Screen session. If there is only one session, the name or ID can be omitted.

### Advanced Technique

- **Scroll back and Copy mode**：
  ```
  Ctrl+a, [
  ```
  You can enter copy mode by this command and scroll by arrow keys or PageUp/PageDown. By `q`, it goes back to normal mode.

- **Manage multiple windows**：
  ```
  Ctrl+a, c    (Make a new window)
  Ctrl+a, n    (Move to the next window)
  Ctrl+a, p    (Move back to the previous window)
  Ctrl+a, "    (Show the window lists.)
  ```

- **Window Split**：
  ```
  Ctrl+a, S    (Horizontal Split)
  Ctrl+a, |    (Vertical Split)
  Ctrl+a, Tab  (Move over splits)
  Ctrl+a, X    (Delete split)
  ```

## SSH

### How to connect to Lab Machine

```
ssh [Your Username]@[hostname of Lab machine]
```

*Connection requires you to use union VPN ([URL](https://union.teamdynamix.com/TDClient/1831/Portal/KB/?CategoryID=10540))

### Authorization with SSH

1. **Generate SSH key pair**
   ```
   ssh-keygen -t rsa -b 4096
   ```
   This command generates a pair of private key (`~/.ssh/id_rsa`) and public key (`~/.ssh/id_rsa.pub`).

2. **Transmit public key to the Lab Machine**
   ```
   ssh-copy-id [Your Username]@[hostname of Lab machine]
   ```
   (You can do thie manualy, but I don't recommened to do it)
   Also, you can specify which key you are using (``` -i ~/.ssh/your_key.pub ```)

3. **Connectino test**
   ```
   ssh [Your Username]@[hostname of Lab machine]
   ```
   If it is correctly set up, you will not need to use password to connect.
   If you attach `-i` on the second step, you should specify which key you are using (``` -i ~/.ssh/your_key ```)

### known_hosts

`~/.ssh/known_hosts` is a file to store the public keys of hots which you have been connected to before. If you correctly follows the steps above, it should be automatically added onto the file.

#### know_hosts manage

In case that the key in the server is changed, You need to delete corresponding row.

```
ssh-keygen -R [hostname of Lab machine]
```

This commands delete the entry of given host from `known_hosts`.

## DiscManagement

### Check disc usage

To chcek file/directory which use up most space in home directory:

```sh
du -ak | sort -nr | more
```

Description:
- `du -ak`: show the disc usage of all files in KB
- `sort -nr`: sort in descending order
- `more`: show more
  - `Space`: show next page
  - `Enter`: show next row
  - `b`: go back one page
  - `q`: quit
  - `/word`: search for `word`

### Other practical command example

#### Only top 10

```sh
du -ak | sort -nr | head -10
```


#### Only under a specific directory
Ex: under the home directory

```sh
du -ak ~/ | sort -nr | more
```

#### Calculate without files with a specific extension
Ex: All files without `.log` files (combination with `find`)

```sh
find . -type f ! -name "*.log" -print0 | xargs -0 du -a | sort -nr | more
```

- `find . -type f ! -name "*.log" -print0`  
  Search for all files without `.log` files ybder current directory and show file names by `\0`
- `xargs -0 du -a`  
  Get file list from the previous command and execute `du -a` (show the disc usage for each file)  
  - `-0`: received as separated by `\0` (Use with `-print0`)

#### Show in in human-readable units
`-h` option allows it to show in KB/MB/GB

```sh
du -ah | sort -hr | more
```

#### Calculate by directory (sub-directory)

```sh
du -sh ./* | sort -hr
```

## Example

### USE Loacl disc

We recommend that you place your experimental configuration and repositories in the `/var/489-02` directory on your local disk, not on a network drive.

Benefit:
- Much faster processing speed (faster than network disk)
- Not affected by home directory space limitations
- Ideal for experiments involving a large number of files

### Sample strategy for long duration experiment

1. Connect to union VPN
2. Connect to Lab machine with ssh
3. Start screen session: `screen -S experiment1`
4. Execute experiment script
5. Detach from screen by Ctrl+a, d
6. Reconnect if necessary: `screen -r experiment1`

## おまけ

To learn more about setting up SSH keys and the detailed functionality of Screen, refer to the `man` page for each command (e.g., `man screen`) or check the detailed online documentation.
