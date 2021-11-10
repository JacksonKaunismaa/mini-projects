var els = document.getElementsByTagName("td");

for (var i=0, max=els.length; i<max; i++) {
				els[i].outerHTML = els[i].outerHTML.split("ђ").join("а").split("ї").join("я").split("Ђ").join("А").split("Ї").join("Я").split("џ").join("у").split("є").join("ю").split("Є").join("Ю").split("Џ").join("У").split("ћ").join("о").split("Ћ").join("О").split("љ").join("е").split("Љ").join("Е").split("ќ").join("и").split("Ќ").join("И");
}


