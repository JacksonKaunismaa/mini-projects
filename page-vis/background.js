console.log("CAN ANYONE HEAR ME?");



function toggleImage(enabled=true){
  // Update icon.
  const name = enabled ? 'cross' : 'watch';
  const title = enabled ? 
							"Page Visibility API is disabled" :
							"Page Visibility API is enabled";
	browser.browserAction.setTitle({title});
	const path = `icons/eyes-${name}-64.jpeg`;
  browser.browserAction.setIcon({ path });
}

function handleClick(){
	console.log("I got a lciked");
	isEnabled = !isEnabled;
	toggleImage(isEnabled);
	browser.tabs.sendMessage(browser.tabs.query({active: true, currentWindow: true})[0].id, {value: isEnabled});
}
	
browser.browserAction.onClicked.addListener(handleClick);
browser.tabs.executeScript({file: "/content_scripts/block.js"});
