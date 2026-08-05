请根据下面的用户请求，选择合适的工具组。

用户请求：

{{routing_task}}

可选工具组：

{{toolset_metadata}}

返回1到3个工具组名称，按重要程度从左到右排列。一个工具组足够时只返回一个。

不需要工具时返回["NO_TOOL"]，NO_TOOL不能和其他工具组同时返回。

只输出JSON字符串数组，不要解释。

示例：

["WEB_RESEARCH"]

["WEB_RESEARCH", "FILE_EDITING"]

["NO_TOOL"]
