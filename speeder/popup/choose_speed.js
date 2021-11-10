function listen(){

	browser.tabs.query({active: true, currentWindow: true})
	.then((the_tabs) => {
		browser.tabs.sendMessage(the_tabs[0].id, {value:"init"}) 
		.then((ret) => {
			switch (ret.val){
				case -1:
					document.getElementById("display").innerHTML = `Video not found!`;
					break;
				default:
					document.getElementById("display").innerHTML = `Speed: ${ret.val}x`;
			}
		})
	});
	
	var evtList = ["click", "keypress"];
	for (evt of evtList){
		document.addEventListener(evt, (e) => {
			

			function ret_bag(error){
				console.error(`Bad thing happened, return failed somehow: ${error}`);
			}

			function click(tabs){
				let thing = document.getElementById("speed").value;
				document.getElementById("speed").value = "";
				browser.tabs.sendMessage(tabs[0].id, {
					value: thing
				}).then((ret) => {
					switch(ret.val){
						case 1:
							document.getElementById("result").style.color = "green";
							document.getElementById("result").innerHTML = `Succesfully updated speed to ${thing}`;
							document.getElementById("display").innerHTML = `Speed ${ret.speed}x`;
							break;
						case 0:
							document.getElementById("result").style.color = "black";
							document.getElementById("result").innerHTML = `Video not found!`;
							break;
						case -1:
							document.getElementById("result").style.color = "red";
							document.getElementById("result").innerHTML = `Invalid speed "${thing}"!`;
							break;
					}
				}).catch(ret_bad);
			}

			function mistake(error){
				console.error(`Mistake happened speed-wise: ${error}`);
			}

			if ((e.key && e.key == "Enter") || (e.target && e.target.classList && e.target.classList.contains("submit"))){
				browser.tabs.query({active: true, currentWindow: true})
					.then(click)
					.catch(mistake);
			}
		});
	}
}

function query_error(error){
	console.error(`query_error is: ${error}`);
}

browser.tabs.executeScript({file: "/content_scripts/content_speed.js"})
.then(listen)
.catch(query_error);
