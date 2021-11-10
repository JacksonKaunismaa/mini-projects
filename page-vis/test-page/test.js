var changed_val = 0;
function sleep(ms){
	return new Promise(resolve => setTimeout(resolve, ms));
}


async function demo() {
	let vis_val = 0;
	let hid_val = 0;

	while (true){
		await sleep(1000);
		if (document.hidden){
			hid_val += 1;
		}
		else {
			vis_val += 1;
		}
		document.getElementById("hid-val").innerHTML = hid_val;
		document.getElementById("vis-val").innerHTML = vis_val;
	}
}

function changed(){
	changed_val += 1;
	document.getElementById("changed-val").innerHTML = changed_val;
}



document.addEventListener("visibilitychange", changed, false);

demo();
