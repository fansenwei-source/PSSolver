# S-22 隔离导入与资格执行更正

f052a4e 全量结果为 5478 passed、2 failed、7 deselected，无 skip。全部日志保留。

分类插件按模块名加载在既有 `--import-mode=importlib`、外部 rootdir 的收集检查中
找不到 `tests`。现由 conftest 按同目录文件加载插件，不增加 Provider 的 sys.path；
插件自行读取 TOML，不依赖工具包可导入。GPU 节点按仓库相对路径规范化，外部 rootdir
也不能绕过前提检查。原有 RC4.4.7 测试不改，新增隔离 rootdir 的 CPU 负向用例。

另一个失败是旧 stationarity 测试检测到分析期间 Git 状态变化。打包日志的时间与测试
重叠，sdist 在该克隆内建立短暂未跟踪的发布树；没有捕获中间 status，故这是有日志
支持的归因而非逐状态重放。该文件独立复测 45/45。最终全量用独占的新克隆，期间
不打包、不启动其他测试。原测试与分析器保持不动，不弱化其失败关闭检查。

本记录向前绑定三个测试支持文件；successor 134、135 和所有旧记录均不改写。
运行时、打包配置、数值实现、benchmark、jobs 和门限均未改变。
