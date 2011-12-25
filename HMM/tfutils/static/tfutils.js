
// TfUtils namespace
var TfUtils = (function() {
    var dictTest = {};
    var dictTask = {};
    
    var jTestResults = $("<span/>");
    var jUpdateNotification = $("<div id='update_notification'" +
				" class='collapsed'/>");

    function getTestHolder() {
	return $("div#test_results");
    }
    
    function getTaskHolder() {
	return $("div#task_results");
    }

    function buildTest(dictTestProperties) {
	var sName = dictTestProperties.name
	var jHolder = getTestHolder();
	var jTest = $("<div class='test'/>");
	var jHeader = $("<div class='test_header'/>");
	var jNameHolder = $("<div class='test_nameholder'/>");
	var jExpandTb = $("<div class='test_expand collapsed'/>");
	var jExpandConsole = $("<div class='test_expand collapsed'/>");
	var jRunTestButton = $("<div class='test_button'/>");
	var jTestContent = $("<div class='test_content collapsed'/>");
	var jPre = $("<pre class='test_pre code'/>");
	jHeader.attr("title",dictTestProperties.description);
	jNameHolder.text(sName.split('.')[1]);
	jHeader.append(jNameHolder);
	jExpandTb.append($("<a href=''/>").text("Show Failure"));
	jHeader.append(jExpandTb);
	jExpandConsole.append($("<a href=''/>").text("Show Output"));
	jHeader.append(jExpandConsole);
	jHeader.append(jRunTestButton);
	jHeader.append($("<div class='clear'/>"));
	jTestContent.append(jPre);
	jTest.append(jHeader);
	jTest.append(jTestContent);
	jHolder.append(jTest);

	function prep() {
	    jRunTestButton.unbind('click');
	    jTest.addClass("running");
	    if (t.showing) {
		jTestContent.slideUp();
		t.showing = null;
	    }
	}

	function cb(event) {
	    prep();
	    runTest(sName);
	}
	jTest.click(cb);
	jRunTestButton.attr("title","Click to run this test.");

	function buildShowText(sKey) {
	    return function(event) {
		if (sKey == t.showing) {
		    jTestContent.slideUp();
		    t.showing = null;
		} else {
		    jPre.text(t[sKey]);
		    if (t.showing === null) {
			jTestContent.slideDown();
		    }
		    t.showing = sKey;
		}
		event.stopPropagation();
		event.preventDefault();
	    }
	}

	function setShowButton(sKey, jButton) {
	    return function(sMsg) {
		t[sKey] = sMsg;
		if (sMsg) {
		    jButton.show();
		} else {
		    jButton.hide();
		}	    
	    }
	}
	jExpandTb.find('a').click(buildShowText("traceback"));
	jExpandConsole.find('a').click(buildShowText("console"));

	var t = {
	    name: sName,
	    j: jTest,
	    jButton: jRunTestButton,
	    result: null,
	    prep: prep,
	    cb: cb,
	    traceback: "",
	    console: "",
	    showing: null,
	    setTraceback: setShowButton("traceback",jExpandTb),
	    setConsole: setShowButton("console", jExpandConsole)
	};

	dictTest[sName] = t;
	setTestResult(sName,dictTestProperties.result);
    }

    function buildTask(dictTaskProperties) {
	var sName = dictTaskProperties.name;
	var sId = dictTaskProperties.id
	var tk = {
	    name: sName,
	    id: sId
	};
	var jTask = $("<div class='task'/>");
	var jHeader = $("<div class='task_header'/>");
	var jDescription = $("<div class='task_description'/>");
	var jTitleHolder = $("<div class='task_title_holder'/>");
	var jTaskContent = $("<div class='task_content collapsed'/>");
	var jError = $("<pre class='task_error collapsed code'/>");
	var jConsole = $("<pre class='task_console collapsed code'/>");
	var jRunHolder = $("<div class='task_run_holder'/>");
	var jUiError = $("<div class='task_ui_error'/>");
	
	jTitleHolder.text(sName);
	
	jRunHolder.append($("<a class='button' href=''/>").text("Run"));
	jHeader.append(jTitleHolder);
	jHeader.append(jRunHolder);

	var jDisplay = null;
	if (dictTaskProperties.type == "graph") {
	    jDisplay = $("<canvas width='480' height='320' class='graph'/>");
	} else if (dictTaskProperties.type == "chart") {
	    jDisplay = $("<div id='" + sId + "_chart'/>");
	}
	jTaskContent.append(jDisplay);

	jTask.append(jHeader);
	jTask.append($("<div class='clear'/>"));
	if (dictTaskProperties.description) {
	    jDescription.text(dictTaskProperties.description);
	    jTask.append(jDescription);
	}
	jTask.append(jUiError);
	jTask.append(jError);
	jTask.append(jConsole);
	jTask.append(jTaskContent);

	tk.display = jDisplay;
	tk.j = jTask;
	function handleErrors(fxn) {
	    return function(json) {
		if (json.console) {
		    jConsole.text(json.console);
		    jConsole.slideDown();
		} else {
		    jConsole.slideUp();
		}
		if (json.tb) {
		    jError.text(json.tb);
		    jError.slideDown();
		} else {
		    jError.slideUp();		
		}
		
		if (json.valid !== false) {
		    var ret = fxn(json.result);
		    if (ret) {
			jTaskContent.slideDown();
		    }
		}
	    }
	}
	tk.cb = (function() {
	    sType = dictTaskProperties.type;
	    if (sType == "graph") {
		return handleErrors(function(result) {
		    var iWidth = jTask.width();
		    var iHeight = Math.floor(iWidth*9.0/16.0);
		    jDisplay.attr("width",iWidth);
		    jDisplay.attr("height", iHeight);
		    var graph = buildSpringyGraph(result);
		    drawSpringyGraph(graph,jDisplay);
		    return graph;
		});
	    } else if (sType == "chart") {
		return handleErrors(function(dictChart) {
		    var iWidth = jTask.width();
		    var iHeight = Math.floor(iWidth*9.0/16.0);
		    dictChart.chart.renderTo = sId + "_chart";
		    dictChart.chart.width = iWidth;
		    dictChart.chart.height = iHeight;
		    var chart = new Highcharts.Chart(dictChart);
		    return chart;
		});
	    }
	})();
	tk.button = jRunHolder.find('a');
	tk.button.click(function(event) {
	    event.preventDefault();
	    jTaskContent.slideUp(500, function(){
		jTask.addClass("running");
		jRunHolder.find('a').hide();
		runTask(sId);
	    });
	});
	tk.addError = function(xhr, sStatus, exn) {
	    var jErrorMsg = $("<div class='task_error_message'/>");
	    var jErrorText = $("<span class='task_error_text'/>");
	    var jDismissHolder = $("<div class='task_error_dismiss' />");
	    var jDismiss = $("<a class='button' href=''>Dismiss</a>");
	    
	    jDismiss.click(function(event) {
		event.preventDefault();
		jErrorMsg.fadeOut(300, jErrorMsg.remove);
	    });
	    jDismissHolder.append(jDismiss);

	    var sMsgBase = "An internal error occurred: ";
	    var sMsg = (sMsgBase + '"' +
			(exn ? exn.message + " (" + sStatus + ")"
			 : sStatus)
			+ '"');
	    jErrorText.text(sMsg);
	    jErrorMsg.append(jErrorText);

	    jErrorMsg.append(jDismissHolder);
	    jErrorMsg.append($("<div class='clear'/>"));
	    jUiError.append(jErrorMsg);
	};

	dictTask[sId] = tk;
	getTaskHolder().append(jTask);
    }

    function buildSpringyGraph(listEdges) {
	var setNode = {};
	var graph = new Graph();
	function addNode(sName) {
	    if (setNode[sName] === undefined) {
		var node = graph.newNode({label: sName});
		setNode[sName] = node;
		return node;
	    }
	    return setNode[sName]
	}
	$.each(listEdges, function(_,listE) {
	    var nodeSrc = addNode(listE[0]);
	    var nodeDest = addNode(listE[1]);
	    var dictProperties = listE[2] || {color: '#FF0000'};
	    graph.newEdge(nodeSrc, nodeDest, dictProperties);
	});
	return graph;
    }

    function drawSpringyGraph(graph,canvas) {
	$(canvas).springy(graph);
    }

    function runTask(sTask) {
	var tk = dictTask[sTask];
	function cleanup() {
	    tk.j.removeClass("running");
	    tk.button.show();
	}
	function handleSuccess(json) {
	    cleanup();
	    tk.cb(json);
	}
	function handleError(xhr,sStatus,exn) {
	    cleanup();
	    tk.addError(xhr,sStatus,exn);
	}
	$.ajax({
	    type: "POST",
	    url: "/task/" + tk.id + "/",
	    data: null,
	    success: handleSuccess, 
	    error: handleError,
	    dataType: "json",
	    timeout: 1000.0 * 60.0 * 40.0 // forty minute timeout
	});
    }

    function runTest(sTest) {
	$("div#run_all").hide();
	$.post("/test/run/", {"tests":sTest}, showTestResults, "json");
    }

    function showTestResults(json) {
	$.each(json, function(ix,d) {
	    var t = dictTest[d.name];
	    setTestResult(d.name, d.results.result,
			  d.results.failures.join("\n\n"),
			  d.results.console);
	});
	$("div#run_all").show();
    }

    function setTestResult(sTest,nResult,sTb,sConsole) {
	var t = dictTest[sTest];
	t.result = nResult;
	var jTest = t.j;
	var jButton = t.jButton;
	setTestButtonColor(jButton, nResult);
	t.setTraceback(sTb);
	t.setConsole(sConsole);
	updateTestResults();
	t.j.removeClass("running");
	jButton.click(t.cb);
    }
    
    function setTestButtonColor(jButton,nResult) {
	$.each(["test_success", "test_failure", "test_unknown"],
	       function(ix,sClass) {jButton.removeClass(sClass);});
	if (nResult === true) {
	    jButton.addClass("test_success");
	} else if (nResult === false) {
	    jButton.addClass("test_failure");
	} else {
	    jButton.addClass("test_unknown");
	}
    }

    function buildRunAllButton() {
	var jButton = $("<a class='button' href=''/>").text("Run All");
	var jHolder = $("<div id='run_all' />");
	jHolder.append(jButton);
	function cb(event) {
	    var listToRun = [];
	    $.each(dictTest, function(_,t) {
		listToRun.push(t.name);
		t.prep();
	    });
	    runTest(listToRun.join(','));
	    event.preventDefault();
	}
	jHolder.find('a').click(cb);
	$("div#run_all_holder").prepend(jHolder)
	    .append($("<div class='clear'/>"));

    }

    function loadInitialData(fxnCb) {
	var dictData = 	{};
	var cFound = 0;
	var cMaxRequests = 2;
	function loadMetadata(json) {
	    dictData.sTaskTitle = json.sTaskTitle;
	    dictData.sTaskSubtitle = json.sTaskSubtitle;
	    loadTasks(json.listTask);
	    cFound++;
	    if (cFound >= cMaxRequests) {
		return fxnCb(dictData);
	    }
	}
	function loadTests(json) {
	    dictData.listTest = json;
	    cFound++;
	    if (cFound >= cMaxRequests) {
		return fxnCb(dictData);
	    }
	}
	function loadTasks(json) {
	    json.sort(function(t1,t2) {
		return t1.priority - t2.priority;
	    });
	    $.each(json,function(_,dictTask) {
		buildTask(dictTask);
	    });
	}
	$.get("/metadata/",null,loadMetadata, "json");
	$.get("/test/load/",null,loadTests, "json");
    }

    function updateTestResults() {
	var c = 0;
	var cTotal = 0;
	$.each(dictTest, function(_,t) {
	    if (t.result) {
		c++;
	    }
	    cTotal++;
	});
	jTestResults.text(" (" + c + "/" + cTotal + ")");
    }

    function setTitle(sTitle, sSubtitle) {
	$(document).attr("title", sTitle + " - " + sSubtitle);
	var jSpanTitle = $("<span/>").text(sTitle);
	var jSpanSubtitle = $("<span/>").text(sSubtitle);
	$("div#right_title").append(jSpanTitle).append($("<br/>"))
						       .append(jSpanSubtitle);
    }

    function setInitialUi(dictData) {
	setTitle(dictData.sTaskTitle, dictData.sTaskSubtitle);
	$.map(dictData.listTest, buildTest);
    }

    function checkForUpdates() {
	function handleSuccess(json) {
	    if (json) {
		var jFound = $("<div class='updates_found'/>");
		var jMsgHolder = $("<span class='update_msg_holder'></span>");
		var jUpdateButton = $("<a href='' class='button'/>");
		jUpdateButton.text("Install Updates");
		jUpdateButton.click(function(event) {
		    event.preventDefault();
		    getUpdates();
		    jUpdateNotification.slideUp();
		    jFound.remove();
		});

		var sMsg = [
		    "Updates are available for this interface."
		].join(" ");
		jMsgHolder.text(sMsg);
		jFound.append(jMsgHolder);
		jFound.append($("<span>&nbsp</span>"));
		jFound.append(jUpdateButton);
		
		jUpdateNotification.append(jFound);
		jUpdateNotification.slideDown(); // we might want this hidden
	    }
	}
	$.ajax({
	    type: "POST",
	    url: "/updates/check/",
	    success: handleSuccess, 
	    dataType: "json"	    
	});
    }

    function getUpdates() {
	function handleSuccess(json) {
	    if (json.success) {
		shutdown();
	    } else {
		alert("Failed to install updates.");
	    }
	}
	$.ajax({
	    type: "POST",
	    url: "/updates/install/",
	    success: handleSuccess,
	    dataType: "json",
	    // 10 minute timeout, just to be sure.
	    timeout: 1000.0 * 60.0 * 10.0
	});
    }

    function shutdown() {
	var jDialog = $("<div class='shutdown_dialog collapsed'/>");
	jDialog.text("Updates were successfully installed. "
		     + "Please restart the interface server.");
	jUpdateNotification.append(jDialog);
	jDialog.dialog({modal:true, draggable:false, resizeable: false});
    }

    function load() {
	$("body").prepend(jUpdateNotification);
	$("div#tabs").tabs();
	loadInitialData(setInitialUi);
	$("span#tab_test_lbl").append(jTestResults);
	buildRunAllButton()
	checkForUpdates();
    }

    return {load: load};
})();

$(document).ready(TfUtils.load);