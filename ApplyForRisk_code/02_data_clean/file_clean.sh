path = "../data_mid"
date = $1
sed -i 's/\"\"/\"/g' $path/data.$date.screen_all.log
sed -i 's/\"\[\[/\[\[/g' $path/data.$date.screen_all.log
sed -i '/^\"$/d' $path/data.$date.screen_all.log
