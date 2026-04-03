"""
最小体积打包脚本
使用 PyInstaller 的优化选项减少最终可执行文件体积
"""
import PyInstaller.__main__
import os
import shutil

def clean_build_dirs():
    """清理构建目录"""
    dirs_to_remove = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"已清理: {dir_name}")

def build_minimal():
    """使用最小体积配置打包"""
    
    # 清理旧的构建文件
    clean_build_dirs()
    
    # PyInstaller 参数 - 优化体积
    args = [
        'main.py',                          # 入口文件
        '--name=OptCNT',                    # 应用名称
        '--onefile',                        # 打包成单个文件
        '--windowed',                       # Windows GUI 应用（无控制台）
        
        # 体积优化选项
        '--strip',                          # 去除符号表
        '--noupx',                          # 不使用 UPX
        
        # 排除不必要的模块（不移除 distutils 和 setuptools，避免冲突）
        '--exclude-module=matplotlib.tests',
        '--exclude-module=numpy.random._examples',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        '--exclude-module=PyQt5',
        '--exclude-module=PyQt6',
        '--exclude-module=PySide2',
        '--exclude-module=PySide6',
        '--exclude-module=tkinter.test',
        '--exclude-module=unittest',
        '--exclude-module=pytest',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=notebook',
        '--exclude-module=sphinx',
        '--exclude-module=pydoc',
        '--exclude-module=email',
        '--exclude-module=http.server',
        '--exclude-module=xmlrpc',
        '--exclude-module=multiprocessing',
        '--exclude-module=concurrent.futures',
        '--exclude-module=ctypes.test',
        '--exclude-module=pip',
        '--exclude-module=wheel',
        '--exclude-module=Cython',
        '--exclude-module=cython',
        '--exclude-module=tkinter.tix',
        '--exclude-module=tkinter.ttk.test',
        '--exclude-module=idlelib',
        '--exclude-module=test',
        '--exclude-module=lib2to3',
        '--exclude-module=ensurepip',
        '--exclude-module=venv',
        '--exclude-module=turtledemo',
        '--exclude-module=turtle',
        
        # 隐藏导入（需要的模块）
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=PIL',
        '--hidden-import=PIL._imagingtk',
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=skimage',
        '--hidden-import=skimage.filters',
        '--hidden-import=skimage.morphology',
        '--hidden-import=matplotlib',
        '--hidden-import=matplotlib.backends.backend_tkagg',
        '--hidden-import=matplotlib.pyplot',
        
        # 收集数据文件
        '--collect-data=skimage',
        '--collect-data=matplotlib',
        
        # 优化级别
        '--optimize=2',
        
        # 清理临时文件
        '--clean',
        
        # 日志级别
        '--log-level=WARN',
    ]
    
    print("开始打包...")
    print(f"参数: {' '.join(args)}")
    
    PyInstaller.__main__.run(args)
    
    print("\n打包完成!")
    
    # 显示输出文件信息
    exe_path = os.path.join('dist', 'OptCNT.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"输出文件: {exe_path}")
        print(f"文件大小: {size_mb:.2f} MB")
    
    # 清理构建目录（保留dist）
    if os.path.exists('build'):
        shutil.rmtree('build')
        print("已清理 build 目录")

if __name__ == '__main__':
    build_minimal()
