console.log("hello from block.jsssss");


(function () {
	if (window.hasRun)
		return;
	window.hasRun = true;
	let isEnabled = false;

	for (evt of ["visibilitychange", "webkitvisibilitychange", "blur"]){
		document.addEventListener(evt, function(event) {
			if (isEnabled){
				console.log("tried to stop an event");
				event.stopImmeadiatePropagation();
			}
		}, true);
	}


	function toggleEnabled(truth_val){
	  isEnabled = truth_val;
	}


	browser.runtime.onMessage.addListener((msg) => { 
		return toggleEnabled(msg.value);
	});
})();
