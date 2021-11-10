(function () {
	
	if (window.hasRun){
		return;
	}

	window.hasRun = true;

	function proc_msg(inpt_msg){
		if (inpt_msg == "init"){
			if (document.getElementsByTagName("video")[0]){
				return Promise.resolve({val: document.getElementsByTagName("video")[0].playbackRate});
			}
			return Promise.resolve({val:-1});
		}
		var inpt = Number(inpt_msg);
		if (!isNaN(inpt) && inpt >= 0 && inpt <= 20){
			if (document.getElementsByTagName("video")[0]){
				document.getElementsByTagName("video")[0].playbackRate = inpt;
				return Promise.resolve({val:1, speed:document.getElementsByTagName("video")[0].playbackRate});
			}
			else {
				console.log("video not found");
				return Promise.resolve({val:0});
			}
				//document.getElementById("result").innerHTML = "Video not found";
		}
		else {
			console.log("invalid speed");
			return Promise.resolve({val:-1});
			//document.getElementById("result").innerHTML = "Failure, speed is invalid";
		}
	}


	browser.runtime.onMessage.addListener((msg) => {
		return proc_msg(msg.value);
	});
})(); 			// call the function instantly


console.log("hello from speeder");
