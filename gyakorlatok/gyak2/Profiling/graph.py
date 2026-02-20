import matplotlib.pyplot as plt

# stages: a mérési lépések nevei
stages = ['Kernel Load/Build', 'Buffer Creation', 'Kernel Run', 'Read Result']

# times: a C programban mért ms értékek
times = [373.00, 15.00, 16.00, 0.00]  # cseréld ki a saját program outputjára

plt.bar(stages, times, color='skyblue')
plt.ylabel('Time (ms)')
plt.title('OpenCL vector addition timing')
plt.show()