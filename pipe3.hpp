#pragma once

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <sys/wait.h>

typedef struct {
    int pid;
    FILE* stdin;
    FILE* stdout;
    FILE* stderr;
} process;

// The popen3 function now takes flags for each stream to control whether to pipe them
int popen3(process* process_obj, char* const* argv, bool pipe_stdin, bool pipe_stdout, bool pipe_stderr) {
    int pipe_stdin_fd[2], pipe_stdout_fd[2], pipe_stderr_fd[2];

    // Conditionally create pipes based on the flags
    if (pipe_stdin && pipe(pipe_stdin_fd) < 0) {
        return -1; // Failed to create pipe for stdin
    }
    if (pipe_stdout && pipe(pipe_stdout_fd) < 0) {
        if (pipe_stdin) {
            ::close(pipe_stdin_fd[0]);
            ::close(pipe_stdin_fd[1]);
        }
        return -1; // Failed to create pipe for stdout
    }
    if (pipe_stderr && pipe(pipe_stderr_fd) < 0) {
        if (pipe_stdin) {
            ::close(pipe_stdin_fd[0]);
            ::close(pipe_stdin_fd[1]);
        }
        if (pipe_stdout) {
            ::close(pipe_stdout_fd[0]);
            ::close(pipe_stdout_fd[1]);
        }
        return -1; // Failed to create pipe for stderr
    }

    const int pid = fork();

    if (pid < 0) { // Fork failed
        // Clean up pipes
        if (pipe_stdin) {
            ::close(pipe_stdin_fd[0]);
            ::close(pipe_stdin_fd[1]);
        }
        if (pipe_stdout) {
            ::close(pipe_stdout_fd[0]);
            ::close(pipe_stdout_fd[1]);
        }
        if (pipe_stderr) {
            ::close(pipe_stderr_fd[0]);
            ::close(pipe_stderr_fd[1]);
        }
        return -1;
    } else if (pid == 0) {  // Child process

        // Close the unused ends of the pipes
        if (pipe_stdin) {
            ::close(pipe_stdin_fd[1]);
            ::dup2(pipe_stdin_fd[0], 0); // Redirect stdin
            ::close(pipe_stdin_fd[0]);
        }
        if (pipe_stdout) {
            ::close(pipe_stdout_fd[0]);
            ::dup2(pipe_stdout_fd[1], 1); // Redirect stdout
            ::close(pipe_stdout_fd[1]);
        }
        if (pipe_stderr) {
            ::close(pipe_stderr_fd[0]);
            ::dup2(pipe_stderr_fd[1], 2); // Redirect stderr
            ::close(pipe_stderr_fd[1]);
        }

        // Execute the command
        if (execvp(argv[0], argv) < 0) {
            // Close the pipes and return failure if execvp fails
            if (pipe_stdin) {
                ::close(pipe_stdin_fd[0]);
                ::close(pipe_stdin_fd[1]);
            }
            if (pipe_stdout) {
                ::close(pipe_stdout_fd[0]);
                ::close(pipe_stdout_fd[1]);
            }
            if (pipe_stderr) {
                ::close(pipe_stderr_fd[0]);
                ::close(pipe_stderr_fd[1]);
            }
            return -1;
        }
    } else { // Parent process

        process_obj->pid = pid;

        // Close the unused ends of the pipes in the parent
        if (pipe_stdin) {
            ::close(pipe_stdin_fd[0]);
        }
        if (pipe_stdout) {
            ::close(pipe_stdout_fd[1]);
        }
        if (pipe_stderr) {
            ::close(pipe_stderr_fd[1]);
        }

        // Open file streams for stdin, stdout, stderr
        if (pipe_stdin) {
            process_obj->stdin = fdopen(pipe_stdin_fd[1], "w");
        } else {
            process_obj->stdin = NULL;  // No stdin piping
        }

        if (pipe_stdout) {
            process_obj->stdout = fdopen(pipe_stdout_fd[0], "r");
        } else {
            process_obj->stdout = NULL; // No stdout piping
        }

        if (pipe_stderr) {
            process_obj->stderr = fdopen(pipe_stderr_fd[0], "r");
        } else {
            process_obj->stderr = NULL; // No stderr piping
        }

        return 0; // Success
    }

    return 0; // Ensure function always returns an int
}

#if 0
int close(process* process_obj) {
    if (process_obj->stdin) fclose(process_obj->stdin);
    if (process_obj->stdout) fclose(process_obj->stdout);
    if (process_obj->stderr) fclose(process_obj->stderr);

    // Wait for the child process to terminate and return the exit status
    int status;
    ::waitpid(process_obj->pid, &status, 0);
    return status;
}
#endif

#if 0
int main() {
    process proc;
    
    // Configure which streams to pipe
    bool pipe_stdin = true;   // Set to true if stdin should be piped
    bool pipe_stdout = true;  // Set to true if stdout should be piped
    bool pipe_stderr = true;  // Set to true if stderr should be piped

    // Command and arguments to execute (e.g., "openssl sha256")
    char* argv[] = {"openssl", "sha256", NULL};

    if (0 != popen3(&proc, argv, pipe_stdin, pipe_stdout, pipe_stderr)) {
        fprintf(stderr, "[error] popen3 failed");
    }

    // Write input data ("hello world") to the child process if stdin is piped
    if (proc.stdin) {
        fprintf(proc.stdin, "hello world\n");
        fclose(proc.stdin);  // Close stdin after writing
    }

    // Read the output from the child process if stdout is piped
    if (proc.stdout) {
        char buf[512];
        buf[0] = '\0'; // Ensure the buffer is empty before reading
        fread(buf, 512, 1, proc.stdout); // Read up to 512 bytes from stdout
        fprintf(stderr, "---- begin of stdout ----\n%s\n---- end of stdout ----\n", buf); // Print stdout data
        fclose(proc.stdout);  // Close stdout
    }

    // Read the error output from the child process if stderr is piped
    if (proc.stderr) {
        char buf[512];
        buf[0] = '\0'; // Clear the buffer before reading stderr
        fread(buf, 512, 1, proc.stderr); // Read up to 512 bytes from stderr
        fprintf(stderr, "---- begin of stderr ----\n%s\n---- end of stderr ----\n", buf); // Print stderr data
        fclose(proc.stderr);  // Close stderr
    }

    // Wait for the child process to terminate and get the exit status
    int status;
    ::waitpid(proc.pid, &status, 0);

    return status;
}
#endif